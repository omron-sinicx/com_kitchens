import copy
import os
from typing import Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.models.x_clip.modeling_x_clip import *
from transformers.models.t5.modeling_t5 import *

from .configuration_xclip_t5 import XCLIPT5Config

logger = logging.get_logger(__name__)

class XCLIPT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XCLIPT5Config
    base_model_prefix = "xclip_t5"
    supports_gradient_checkpointing = True
    
    # load_tf_weights = load_tf_weights_in_t5
    # base_model_prefix = "transformer"
    # is_parallelizable = True
    # supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""

        # XCLIP
        factor = self.config.initializer_factor
        if isinstance(module, XCLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, XCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, XCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, XCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, XCLIPModel):
            factor = self.config.initializer_factor
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )
            nn.init.normal_(module.prompts_visual_projection, mean=0.0, std=module.vision_embed_dim**-0.5 * factor)
        elif isinstance(module, XCLIPMultiframeIntegrationTransformer):
            nn.init.normal_(module.position_embedding, std=self.config.initializer_factor)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            if module.bias is not None:
                module.bias.data.zero_()

        # T5
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        # elif isinstance(module, T5ClassificationHead):
        #     module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
        #     if hasattr(module.dense, "bias") and module.dense.bias is not None:
        #         module.dense.bias.data.zero_()
        #     module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
        #     if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
        #         module.out_proj.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))


    def _set_gradient_checkpointing(self, module, value=False):
        # XCLIP
        if isinstance(module, (XCLIPEncoder, XCLIPVisionEncoder)):
            module.gradient_checkpointing = value
        # T5
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    # T5
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class XCLIPT5Model(XCLIPT5PreTrainedModel):
    config_class = XCLIPT5Config

    # T5
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: XCLIPT5Config):
        super().__init__(config)

        # XCLIP
        # if not isinstance(config.text_config, XCLIPTextConfig):
        #     raise ValueError(
        #         "config.text_config is expected to be of type XCLIPTextConfig but is of type"
        #         f" {type(config.text_config)}."
        #     )

        if not isinstance(config.vision_config, XCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type XCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # text_config = config.text_config
        vision_config = config.vision_config
        t5_config = config.t5_config

        self.projection_dim = config.projection_dim
        # self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # self.text_model = XCLIPTextTransformer(text_config)
        self.vision_model = XCLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # self.prompts_visual_layernorm = nn.LayerNorm(self.vision_embed_dim, eps=config.vision_config.layer_norm_eps)
        # self.prompts_visual_projection = nn.Parameter(torch.randn(self.vision_embed_dim, self.projection_dim))

        mit_config = copy.deepcopy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)

        # self.prompts_generator = XCLIPPromptGenerator(config)

        # T5
        self.model_dim = t5_config.d_model

        self.shared = nn.Embedding(t5_config.vocab_size, t5_config.d_model)

        # encoder_config = copy.deepcopy(config)
        # encoder_config.is_decoder = False
        # encoder_config.use_cache = False
        # encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(t5_config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = t5_config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(t5_config.d_model, t5_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # # Model parallel
        # self.model_parallel = False
        # self.device_map = None

    # def parallelize(self, device_map=None):
    #     warnings.warn(
    #         "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
    #         " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
    #         " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
    #         " 0, 'encoder.block.1': 1, ...}",
    #         FutureWarning,
    #     )
    #     self.device_map = (
    #         get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.encoder.block))
    #     self.encoder.parallelize(self.device_map)
    #     self.decoder.parallelize(self.device_map)
    #     self.model_parallel = True

    # def deparallelize(self):
    #     warnings.warn(
    #         "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
    #         FutureWarning,
    #     )
    #     self.encoder.deparallelize()
    #     self.decoder.deparallelize()
    #     self.encoder = self.encoder.to("cpu")
    #     self.decoder = self.decoder.to("cpu")
    #     self.model_parallel = False
    #     self.device_map = None
    #     torch.cuda.empty_cache()

    # XCLIP
    def get_video_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:

        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[0]
        video_embeds = self.visual_projection(video_embeds)

        # video_embeds = vision_outputs[1]
        # video_embeds = self.visual_projection(video_embeds)

        # cls_features = video_embeds.view(batch_size, num_frames, -1)

        # mit_outputs = self.mit(
        #     cls_features,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # video_embeds = mit_outputs[1]

        return video_embeds

    # T5
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    # def get_encoder(self):
    #     return self.encoder

    def get_decoder(self):
        return self.decoder

    # def _prune_heads(self, heads_to_prune):
    #     """
    #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #     class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)
    
    # XCLIP_T5
    def forward(
        self,
        # XCLIP
        # input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # T5
        # input_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        # # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        # vision_outputs = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # video_embeds = vision_outputs[1]
        # video_embeds = self.visual_projection(video_embeds)

        # cls_features = video_embeds.view(batch_size, num_frames, -1)

        # mit_outputs = self.mit(
        #     cls_features,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # video_embeds = mit_outputs[1]

        # # img_features = vision_outputs[0][:, 1:, :]
        # # img_features = self.prompts_visual_layernorm(img_features)
        # # img_features = img_features @ self.prompts_visual_projection
        # # img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
        # # img_features = img_features.mean(dim=1, keepdim=False)

        # # text_outputs = self.text_model(
        # #     input_ids=input_ids,
        # #     attention_mask=attention_mask,
        # #     position_ids=position_ids,
        # #     output_attentions=output_attentions,
        # #     output_hidden_states=output_hidden_states,
        # #     return_dict=return_dict,
        # # )

        # # text_embeds = text_outputs[1]
        # # text_embeds = self.text_projection(text_embeds)

        # # text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        # # text_embeds = text_embeds + self.prompts_generator(text_embeds, img_features)

        # # normalized features
        # video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        # # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # # # cosine similarity as logits
        # # logit_scale = self.logit_scale.exp()
        # # logits_per_video = torch.einsum("bd,bkd->bk", video_embeds, logit_scale * text_embeds)
        # # logits_per_text = logits_per_video.T

        # # loss = None
        # # if return_loss:
        # #     loss = x_clip_loss(logits_per_text)

        # # if not return_dict:
        # #     output = (logits_per_video, logits_per_text, text_embeds, video_embeds, text_outputs, vision_outputs)
        # #     return ((loss,) + output) if loss is not None else output

        # # return XCLIPOutput(
        # #     loss=loss,
        # #     logits_per_video=logits_per_video,
        # #     logits_per_text=logits_per_text,
        # #     text_embeds=text_embeds,
        # #     video_embeds=video_embeds,
        # #     text_model_output=text_outputs,
        # #     vision_model_output=vision_outputs,
        # #     mit_output=mit_outputs,
        # # )

        # T5
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # # Convert encoder inputs in embeddings if needed
            # encoder_outputs = self.encoder(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     inputs_embeds=inputs_embeds,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            # move vision encoder here.
            encoder_outputs = self.get_video_features(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            hidden_states = encoder_outputs
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            hidden_states = encoder_outputs[0]

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

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


__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

if __name__ == "__main__":
    # https://huggingface.co/docs/transformers/main/model_doc/xclip#transformers.XCLIPModel.forward.example
    import av
    import torch
    import numpy as np

    from transformers import AutoProcessor, AutoModel
    from huggingface_hub import hf_hub_download
    from .processing_xclip_t5 import XCLIPT5Processor

    np.random.seed(0)

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    file_path = hf_hub_download(
        repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    )
    container = av.open(file_path)

    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    processor = XCLIPT5Processor.from_baseclass(AutoProcessor.from_pretrained("microsoft/xclip-base-patch32"))
    config = XCLIPT5Config(
        pad_token_id=processor.tokenizer.pad_token_id,
        decoder_start_token_id=processor.tokenizer.pad_token_id,
    )
    model = XCLIPT5Model(config)

    inputs = processor(
        text=["playing sports", "eating spaghetti", "go shopping"],
        videos=list(video),
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    print(outputs)