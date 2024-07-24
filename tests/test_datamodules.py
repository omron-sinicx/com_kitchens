from pathlib import Path

import pytest
import torch

from com_kitchens.data.comkitchens_datamodule import ComKitchensDataModule
from com_kitchens.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()
    print(dm.hparams)

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [4, 8])
def test_comkitchens_datamodule(batch_size):
    data_path = "/workspace/project/dataset/com-kitchens"

    dm = ComKitchensDataModule(
        data_dir=data_path,
        frame_dir=data_path,
        batch_size=batch_size,
    )
    dm.prepare_data()
    print(dm.hparams)

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_path, "train").exists()
    assert Path(data_path, "val").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    print(num_datapoints)

    batch = next(iter(dm.train_dataloader()))
    pairs_text, pairs_mask, pairs_segment, video, video_mask = batch
    assert (
        len(pairs_text)
        == len(pairs_mask)
        == len(pairs_segment)
        == len(video)
        == len(video_mask)
        == batch_size
    )
    assert len(pairs_text[0]) <= dm.hparams.max_words
    assert len(video[0]) <= dm.hparams.max_frames
