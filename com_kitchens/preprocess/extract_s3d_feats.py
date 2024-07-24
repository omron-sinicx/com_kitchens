import argparse
import math
import os

import ffmpeg
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from com_kitchens.data.hf_datasets import comkitchens
from com_kitchens.preprocess.model import get_model

COKKITCHEN_DATASET_PATH = comkitchens.__path__[0]

# Hard coded values
FRAMERATE_DICT = {"2d": 1, "3d": 24, "s3dg": 30, "raw_data": 16}
SIZE_DICT = {"2d": 224, "3d": 112, "s3dg": 224, "raw_data": 224}
CENTERCROP_DICT = {"2d": False, "3d": True, "s3dg": True, "raw_data": True}
FEATURE_LENGTH = {"2d": 2048, "3d": 2048, "s3dg": 1024, "raw_data": 1024}


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
        self,
        df,
        framerate=1,
        size=224,
        centercrop=True,
    ):
        self.csv = df
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        num, denum = video_stream["avg_frame_rate"].split("/")
        frame_rate = int(num) / int(denum)
        return height, width, frame_rate

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.csv["video_path"].values[idx]
        output_file = self.csv["feature_path"].values[idx]
        video_start = self.csv["video_start"].values[idx]
        video_end = self.csv["video_end"].values[idx]

        if not os.path.isfile(video_path):
            print(f"Video does not exists: {video_path}")
            video = th.zeros(1)
        elif os.path.isfile(output_file):
            print(f"Feature file already exists: {output_file}")
            video = th.zeros(1)
        else:
            print(f"Decoding video: {video_path}")
            try:
                h, w, fr = self._get_video_dim(video_path)
            except:
                print(f"ffprobe failed at: {video_path}")
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }

            if fr < 1:
                print(f"Corrupted Frame Rate: {video_path}")
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }
            height, width = self._get_output_dim(h, w)

            try:
                cmd = (
                    ffmpeg.input(video_path)
                    .trim(start_frame=video_start, end_frame=video_end)
                    .filter("fps", fps=self.framerate)
                    .filter("scale", width, height)
                )
                if self.centercrop:
                    x = int((width - self.size) / 2.0)
                    y = int((height - self.size) / 2.0)
                    cmd = cmd.crop(x, y, self.size, self.size)
                out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                    capture_stdout=True, quiet=True
                )
            except:
                print(f"ffmpeg error at: {video_path}")
                return {
                    "video": th.zeros(1),
                    "input": video_path,
                    "output": output_file,
                }

            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size

            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.from_numpy(video.astype("float32"))
            video = video.permute(0, 3, 1, 2)

        return {
            "video": video,
            "input": video_path,
            "output": output_file,
        }


class RandomSequenceSampler(Sampler):
    def __init__(self, n_sample, seq_len):
        self.n_sample = n_sample
        self.seq_len = seq_len

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        np.random.shuffle(idx)
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)


class Normalize:
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing:
    def __init__(self, type, FRAMERATE_DICT):
        self.type = type
        self.FRAMERATE_DICT = FRAMERATE_DICT
        if type == "2d":
            self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif type == "3d":
            self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
        elif type == "s3dg":
            pass
        elif type == "raw_data":
            pass

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        if self.type == "2d":
            tensor = tensor / 255.0
            tensor = self.norm(tensor)
        elif self.type == "3d":
            tensor = self._zero_pad(tensor, 16)
            tensor = self.norm(tensor)
            tensor = tensor.view(-1, 16, 3, 112, 112)
            tensor = tensor.transpose(1, 2)
        elif self.type == "s3dg":
            tensor = tensor / 255.0
            tensor = self._zero_pad(tensor, self.FRAMERATE_DICT[self.type])
            # To Batch= T x 3 x H x W
            tensor_size = tensor.size()
            tensor = tensor.view(
                -1, self.FRAMERATE_DICT[self.type], 3, tensor_size[-2], tensor_size[-1]
            )
            # To Batch x 3 x T x H x W
            tensor = tensor.transpose(1, 2)
        elif self.type == "raw_data":
            tensor = tensor / 255.0
            tensor = self._zero_pad(tensor, self.FRAMERATE_DICT[self.type])
            # To Batch= T x 3 x H x W
            tensor_size = tensor.size()
            tensor = tensor.view(
                -1, self.FRAMERATE_DICT[self.type], 3, tensor_size[-2], tensor_size[-1]
            )
            # To Batch x 3 x T x H x W
            tensor = tensor.transpose(1, 2)

        return tensor


def hf_to_pd(
    hf_ds, feature_dir, video_name="front_compressed.mp4", feat_name="front_compressed.npy"
):
    assert feature_dir is not None

    def get_start(aps):
        before_frames = [
            int(b["frame"])
            for xnode in aps
            for ynode in xnode["nodes"]
            for b in ynode["meta_info"]["before"]
        ]
        return min(before_frames)

    def get_end(aps):
        after_frames = [
            int(b["frame"])
            for xnode in aps
            for ynode in xnode["nodes"]
            for b in ynode["meta_info"]["after"]
        ]
        return max(after_frames)

    df = pd.concat([ds.to_pandas() for ds in hf_ds.values()], ignore_index=True)

    df["video_path"] = df["path"].map(lambda x: os.path.join(os.path.dirname(x), video_name))
    df["feature_path"] = df.apply(
        lambda x: os.path.join(feature_dir, x["recipe_id"], x["kitchen_id"], feat_name), axis=1
    )
    df["video_start"] = df["actions_by_person"].map(get_start)
    df["video_end"] = df["actions_by_person"].map(get_end)

    return df[["video_path", "feature_path", "video_start", "video_end"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Easy video feature extractor")

    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/workspace/com_kitchens/tmp/comkitchen_feature_s3d",
        help="directory to save feature data",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for extraction")
    parser.add_argument(
        "--half_precision",
        type=int,
        default=1,
        help="whether to output half precision float or not",
    )
    parser.add_argument(
        "--num_decoding_thread",
        type=int,
        default=2,
        help="number of parallel threads for video decoding",
    )
    parser.add_argument(
        "--l2_normalize", type=int, default=0, help="whether to l2 normalize the output feature"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=1024, help="output video feature dimension"
    )

    parser.add_argument(
        "--type",
        type=str,
        default="s3dg",
        help="CNN type",
        choices=["2d", "3d", "s3dg", "raw_data"],
    )
    parser.add_argument(
        "--s3d_model_path",
        type=str,
        default="cache/s3d_howto100m.pth",
        help="S3GD model path",
    )

    args = parser.parse_args()

    hf_datasets = load_dataset(COKKITCHEN_DATASET_PATH, "raw")
    df = hf_to_pd(hf_datasets, feature_dir=args.feature_dir)

    dataset = VideoLoader(
        df,
        framerate=FRAMERATE_DICT[args.type],
        size=SIZE_DICT[args.type],
        centercrop=CENTERCROP_DICT[args.type],
    )

    n_dataset = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_decoding_thread,
        sampler=RandomSequenceSampler(n_dataset, 10) if n_dataset > 10 else None,
    )

    preprocess = Preprocessing(args.type, FRAMERATE_DICT)
    model = get_model(args)
    model = model.cuda()

    with th.no_grad():
        for k, data in enumerate(loader):
            input_file = data["input"][0]
            output_file = data["output"][0]

            if len(data["video"].shape) > 3:
                print(f"Computing features of video {k + 1}/{n_dataset}: {input_file}")
                video = data["video"].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                    for i in tqdm(range(n_iter)):
                        min_ind = i * args.batch_size
                        max_ind = (i + 1) * args.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = model(video_batch)
                        if args.l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if args.half_precision:
                        features = features.astype("float16")

                    output_dir = os.path.dirname(output_file)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    np.save(output_file, features)
                    print(f"Saved feature to {output_file}")
            else:
                print(f"Video {input_file} already processed.")
