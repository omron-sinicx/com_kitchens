import os

import numpy as np
import torch
import torch as th
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


class S3DFeatureExtractor:
    def __init__(
        self,
        max_frames=12,
    ):
        # self.resolution = resolution
        self.max_frames = max_frames
        # self.transform = self._transform(self.resolution)
        # self.frame_slice_pos = frame_slice_pos
        # self.frame_order = frame_order

    def extract_features(self, frame_dir, begin_frame, end_frame):
        # load features
        if os.path.exists(os.path.join(frame_dir, "front_compressed.npy")):
            feats = np.load(os.path.join(frame_dir, "front_compressed.npy"))
        else:
            # DUMMY DATA
            feats = np.load(
                "/workspace/com_kitchens/tmp/comkitchen_feature_s3d/329063/77/front_compressed.npy"
            )

        # print("S3DFeatureExtractor feats.shape (before): ", feats.shape)
        begin_sec = begin_frame // 30
        end_sec = end_frame // 30
        feats = feats[0:end_sec]

        # if shorter, pad with zeros
        if feats.shape[0] < self.max_frames:
            feats = np.pad(
                feats,
                ((0, self.max_frames - feats.shape[0]), (0, 0)),
                "constant",
                constant_values=0,
            )
        else:
            feats = feats[: self.max_frames]

        # print("S3DFeatureExtractor feats.shape (after): ", feats.shape)

        # return dummy data for extra data resources
        # if "99999999" in frame_dir:
        #     video = np.random.rand(1, self.max_frames, 1, 3, 224, 224)
        #     mask = np.ones((1, self.max_frames), dtype=np.longlong)
        #     return video, mask

        mask = np.ones((1, self.max_frames), dtype=np.longlong)
        # print("S3DFeatureExtractor feats.shape: ", feats.shape)
        # print("S3DFeatureExtractor mask.shape: ", mask.shape)

        return feats, mask

    def __call__(self, frame_dir, begin_frame, end_frame):
        return self.extract_features(frame_dir, begin_frame, end_frame)
