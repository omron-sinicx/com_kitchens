#!/usr/bin/env python

import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, Tensor

from typing import Any
from lightning import LightningModule

from transformers import ViTModel, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from torchmetrics import MeanMetric

from com_kitchens import utils
from com_kitchens.models.components.transformer import (
    PositionalEncoding,
    get_pad_mask,
    get_pad_attn_mask,
)

log = utils.get_pylogger(__name__)

class DenseVideoCaptioningModule(LightningModule):
    def __init__(
        self,
        vis_enc_name: str = 'google/vit-base-patch16-224-in21k',
        dec_name: str = 'megagonlabs/t5-base-japanese-web',
        d_model: int = 768,
        dropout: float = 0.2,
        max_input_len: int = 5000,
        vocab_size: int = 32100,
        feat_mask_id: int = 1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.enc_pos_encoding = self._get_enc_pos_encoding()
        self.encoder = self._get_encoder()
        self.decoder, self.lm_head = self._get_decoder()

        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def _get_enc_pos_encoding(self):
        pos_emb = PositionalEncoding(
            d_model = self.hparams.d_model,
            dropout = self.hparams.dropout,
            max_len = self.hparams.max_input_len,
        )
        return pos_emb

    def _get_encoder(self):
        vit = ViTModel.from_pretrained(self.hparams.vis_enc_name)
        enc = vit.encoder
        return enc

    def _get_decoder(self):
        t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.dec_name)
        dec = t5.decoder
        lm_head = t5.lm_head

        # expand embedding for spatial tokens
        if self.hparams.vocab_size > t5.config.vocab_size:
            embed_dim = dec.config.d_model
            additional_tokens = self.hparams.vocab_size - t5.config.vocab_size

            dec.num_embeddings = self.hparams.vocab_size
            dec.embed_tokens.weight = nn.Parameter(
                torch.cat((dec.embed_tokens.weight, torch.randn(additional_tokens, embed_dim)))
            )

            lm_head.weight = nn.Parameter(
                torch.cat((lm_head.weight, torch.randn(additional_tokens, embed_dim)))
            )
            # bias=False for lm_head in T5
        
        return dec, lm_head
    
    def get_mask_mask(self, x, y=None, padding_id=0, masking_seq=False):

        if y is None:
            y = x

        pad_mask = get_pad_mask(x, padding_id)

        attn_mask = get_pad_attn_mask(
            x, 
            y, 
            masking_seq=masking_seq, 
            padding_idx=padding_id
        )

        return pad_mask, attn_mask

    def forward(
        self, 
        features: Tensor,
        feature_masks: Tensor, 
        captions: Tensor,
        caption_attention_mask: Tensor,
    ):
        """
        Arguments:
          features - Tensor (batch_size x spatial_size x d_model)
          captions - Tensor (batch_size x token_len)
        """
        features = self.enc_pos_encoding(features.transpose(0, 1)).transpose(0, 1)

        _, enc_attn_mask = self.get_mask_mask(
            feature_masks, 
            padding_id=self.hparams.feat_mask_id
        )
        dec_attn_mask, dec_enc_attn_mask = self.get_mask_mask(
            captions,
            ~feature_masks,
            padding_id=self.hparams.pad_token_id,
            masking_seq=True,
        )

        enc_out = self.encoder(features, enc_attn_mask)        
        encoder_outputs = enc_out['last_hidden_state']

        decoder_outputs = self.decoder(
            input_ids=captions,
            attention_mask=dec_attn_mask,
            encoder_attention_mask=dec_enc_attn_mask,

            encoder_hidden_states=encoder_outputs,
        )

        # logit
        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.decoder.config.d_model**-0.5)
        lm_logits = self.lm_head(sequence_output)

        # loss
        loss = self.criterion(lm_logits.view(-1, lm_logits.size(-1)), captions.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch):
        """
        batch:
          input_ids
          attention_mask
          token_type_ids
          video
          video_mask
        """
        videos = batch['video']
        video_masks = batch['video_mask']

        captions = batch['input_ids']
        caption_masks = batch['attention_mask']

        model_outputs = self.forward(videos, video_masks, captions, caption_masks)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.model_step(batch)

        self.train_loss(loss)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

if __name__ == '__main__':
    import torch
    from transformers.models.vit.modeling_vit import ViTEncoder
    from com_kitchens.data.components.clip_tokenizers import JapaneseCLIPTokenizerFast
    from com_kitchens.data.comkitchens_datamodule import COMKitchensDataModule

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    time_tokens = [f"<{i}>" for i in range(100)]
    tokenizer = JapaneseCLIPTokenizerFast.from_pretrained(
        "rinna/japanese-roberta-base",
        additional_special_tokens=time_tokens,
    )

    datamodule = COMKitchensDataModule(
        task="dvc",
        dat_files={
            "train": "./data/main/train.dat",
            "validation": "./data/main/val.dat",
            "test": "./data/main/test.dat",
        },
        recipe_dir="./data/main",
        video_dir="./data/clip_226_1fps",
        tokenizer=tokenizer,
        num_workers=8,
        batch_size=4,
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        break
    batch = batch.to(device)

    model = DenseVideoCaptioningModule(
        vocab_size=len(tokenizer.vocab),
    )
    model = model.half()
    model = model.to(device)

    loss = model.model_step(batch)

    print(loss)