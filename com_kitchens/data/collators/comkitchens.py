from typing import Any, Dict, List

import numpy as np
import torch
from transformers import DataCollatorWithPadding


class COMKitchensDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
    ):
        super().__init__(tokenizer=tokenizer)

    def pad_video_feats(self, features):
        video_feats = [feat["video"] for feat in features]

        max_len = max([feat.shape[0] for feat in video_feats])
        feat_dim = video_feats[0].shape[1]

        # all features have the same shape
        if all([feat.shape[0] == max_len for feat in video_feats]):
            return features

        for feat in features:
            video_feat = feat["video"]

            padded = np.zeros((max_len, feat_dim)).astype(video_feat.dtype)
            padded[: video_feat.shape[0], : video_feat.shape[1]] = video_feat
            feat["video"] = padded

            padded_mask = np.ones((max_len,))
            padded_mask[: video_feat.shape[0]] = 0
            feat["video_mask"] = padded_mask

        return features

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # you may see the UserWarning:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        # which is fixed in https://github.com/huggingface/transformers/pull/24772

        features = self.pad_video_feats(features)

        batch = super().__call__(features)
        return batch
