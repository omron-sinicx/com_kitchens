import os

import numpy as np
import torch
import torch as th
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


class RGBExtractor:
    def __init__(
        self,
        resolution=224,
        max_frames=12,
        frame_slice_pos=0,
        frame_order=0,  # 0: ordinary order; 1: reverse order; 2: random order.
    ):
        self.resolution = resolution
        self.max_frames = max_frames
        # self.transform = self._transform(self.resolution)
        self.frame_slice_pos = frame_slice_pos
        self.frame_order = frame_order

        self.transform = Compose(
            [
                Resize(resolution, interpolation=Image.BICUBIC),
                CenterCrop(resolution),
                self.load_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

        assert self.frame_order in (0, 1, 2)

    def load_rgb(self, im):
        return im.convert("RGB")

    def process_raw_data(self, feats: torch.Tensor):
        tensor_size = feats.size()
        n_channel, weidth, height = tensor_size[-3], tensor_size[-2], tensor_size[-1]

        tensor = feats.view(-1, 1, n_channel, weidth, height)

        return tensor

    def slice_frames(self, frame_feat):
        if self.max_frames < frame_feat.shape[0]:
            if self.frame_slice_pos == 0:
                sliced = frame_feat[: self.max_frames, ...]
            elif self.frame_slice_pos == 1:
                sliced = frame_feat[-self.max_frames :, ...]
            else:
                sample_indx = np.linspace(
                    0, frame_feat.shape[0] - 1, num=self.max_frames, dtype=int
                )
                sliced = frame_feat[sample_indx, ...]
        else:
            sliced = frame_feat

        return sliced

    def order_frames(self, raw_video_data):
        if self.frame_order == 0:
            pass
        elif self.frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif self.frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

    def extract_features(self, frame_dir, begin_frame, end_frame):
        # return dummy data for extra data resources
        if "99999999" in frame_dir:
            video = np.random.rand(1, self.max_frames, 1, 3, 224, 224)
            mask = np.ones((1, self.max_frames), dtype=np.longlong)
            return video, mask

        # load paths to all frames
        image_paths = []
        for root, dirs, files in os.walk(frame_dir):
            for dir in dirs:
                frame_dir_by_minute = os.path.join(root, dir)
                image_paths.extend(
                    [
                        os.path.join(frame_dir_by_minute, file)
                        for file in os.listdir(frame_dir_by_minute)
                    ]
                )

        trimmed = image_paths[begin_frame:end_frame]
        # TODO: when len(trimmed) % (self.max_frames-1) == 0,
        # `sampled` has only (self.max_frames-1) frames and ignore the last frame.
        sampled = trimmed[:: len(trimmed) // (self.max_frames)]

        # process RGB feats (torch)
        feats = torch.stack([self.transform(Image.open(im)) for im in sampled])
        assert len(feats) > 3, f"video path: {frame_dir} error."

        feats = self.process_raw_data(feats)
        feats = self.slice_frames(feats)
        feats = self.order_frames(feats)

        n_channel, weidth, height = feats.shape[-3], feats.shape[-2], feats.shape[-1]
        feats = feats.view(1, self.max_frames, 1, n_channel, weidth, height)

        mask = np.ones((1, self.max_frames), dtype=np.longlong)

        return feats.numpy().astype(np.float64), mask

    def __call__(self, frame_dir, begin_frame, end_frame):
        return self.extract_features(frame_dir, begin_frame, end_frame)
