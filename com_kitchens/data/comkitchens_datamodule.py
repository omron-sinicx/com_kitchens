from typing import Any, Dict, List, Optional, Tuple

from datasets import Split, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, distributed
from transformers.models.clip.tokenization_clip_fast import CLIPTokenizerFast

from com_kitchens.data.collators.comkitchens import COMKitchensDataCollatorWithPadding
from com_kitchens.data.components.datasets import DVCDataset, RetrievalDataset
from com_kitchens.data.hf_datasets import comkitchens
from com_kitchens.data.hf_datasets.comkitchens import COMKITCHENS_TASKS

COKKITCHEN_DATASET_PATH = comkitchens.__path__[0]

TASK_DATASETS = {"retrieval": RetrievalDataset, "dvc": DVCDataset}


class COMKitchensDataModule(LightningDataModule):
    OPTIONAL_LOAD_DATA_KW = [
        "dat_files",
        "recipe_dir",
        "video_dir",
        "lang",
        "ap_segment",
        "sep",
        "ignore_absent",
        "test_frame_ratios",
    ]

    def __init__(
        self,
        task: str,
        tokenizer,
        video_extractor=None,
        dat_files=None,
        recipe_dir=None,
        video_dir=None,
        lang=None,
        ap_segment=None,
        sep=None,
        ignore_absent=None,
        max_words=32,
        batch_size=32,
        num_workers: int = 0,
        pin_memory: bool = False,
        test_frame_ratios=[0.25, 0.5, 0.75, 1.0],
    ):
        assert task in COMKITCHENS_TASKS

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # UniVL -> BertTokenizer, CLIP4Clip, X-CLIP -> CLIPTokenizerFast
        self.tokenizer = None
        self.video_extractor = None
        self.dataset = None
        self.collator_fn = None

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def setup(self, stage: Optional[str] = None):
        # things to do on every process in DDP
        # load data, set variables, etc...

        if not self.tokenizer:
            self.tokenizer = self.hparams.tokenizer

        if not self.video_extractor:
            self.video_extractor = self.hparams.video_extractor

        if not self.collator_fn:
            self.collator_fn = COMKitchensDataCollatorWithPadding(tokenizer=self.tokenizer)

        if not self.dataset:
            kwargs = {}
            for keyword in self.OPTIONAL_LOAD_DATA_KW:
                if (value := getattr(self.hparams, keyword, None)) is not None:
                    kwargs[keyword] = value

            dataset = load_dataset(
                COKKITCHEN_DATASET_PATH,
                name=self.hparams.task,
                cache_dir=None,
                **kwargs,
            )

            # tokenize and extract video
            kwargs = {}
            if self.video_extractor:
                kwargs['transform'] = self.video_extractor.__call__

            self.dataset = {
                subset: TASK_DATASETS[self.hparams.task](
                    hf_ds=data,
                    tokenize=self.tokenizer.__call__,
                    max_words=self.hparams.max_words,
                    **kwargs,
                )
                for subset, data in dataset.items()
            }

    def train_dataloader(self):
        # return train dataloader
        return DataLoader(
            self.dataset[Split.TRAIN],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
            drop_last=True,  # drop the last incomplete batch
        )

    def val_dataloader(self):
        # return validation dataloader
        return DataLoader(
            self.dataset[Split.VALIDATION],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # return test dataloader
        return DataLoader(
            self.dataset[Split.TEST],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    # Test for UniVL
    from transformers import AutoTokenizer
    import datasets

    # Test for CLIP4Clip, X-CLIP
    from com_kitchens.data.components.rgb_extractor import RGBExtractor

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    video_extractor = RGBExtractor()

    datamodule = COMKitchensDataModule(
        task="retrieval",
        tokenizer=tokenizer,
        video_extractor=video_extractor,
        num_workers=8,
        lang="en",
        ap_segment="begin",
        batch_size=40,
        sep=" ",
        ignore_absent=True,
        dat_files= {
            datasets.Split.TRAIN: 'data/main/train.dat',
            datasets.Split.VALIDATION: 'data/main/val.dat',
            datasets.Split.TEST: 'data/main/val.dat',
        }
    )

    datamodule.prepare_data()
    datamodule.setup()

    dup_ratios = []
    from tqdm import tqdm

    for _ in tqdm(range(10)):
        for batch in datamodule.train_dataloader():
            recipe_ids = batch["recipe_id"]
            kitchen_ids = batch["kitchen_id"]
            count = 0  # count duplication
            for i in range(len(recipe_ids)):
                for j in range(len(recipe_ids)):
                    if recipe_ids[i] == recipe_ids[j]:
                        count += 1
            assert count >= len(recipe_ids)
            dup_ratios.append(count / (len(recipe_ids) * (len(recipe_ids))))

    print("average duplication ratio: ", sum(dup_ratios) / len(dup_ratios))
