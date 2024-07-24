from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        tokenize: Optional[Callable],
        transform: Optional[Callable],
        max_words: int,
    ) -> None:
        super().__init__()

        self.ds = hf_ds

        self.tokenize = tokenize
        self.transform = transform

        self.max_words = max_words

    def tokenize_and_extract(self, item):
        """
        item -  {
                    "id": "1126066/66/1-1",
                    "text": "place the chicken tenders on a small",
                    "begin": 1490,
                    "end": 1720,
                    "frame_dir": "./data/frames/test/1126066/66"
                }
        """
        # tokenization
        net_input = self.tokenize(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_words,
        )

        # segment id
        net_input["token_type_ids"] = [0 for _ in range(len(net_input["input_ids"]))]

        # video extraction
        video, video_mask = self.transform(item["frame_dir"], item["begin"], item["end"])
        net_input["video"] = video
        net_input["video_mask"] = video_mask
        net_input["is_query"] = item["is_query"]
        net_input["is_pool"] = item["is_pool"]

        # id
        net_input.update(
            {
                "recipe_id": item["recipe_id"],
                "kitchen_id": item["kitchen_id"],
                "ap_id": int(item["ap_id"]),
            }
        )

        return net_input

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        return self.tokenize_and_extract(item)


class DVCDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        tokenize: Optional[Callable],
        max_words: int,
        time_bins: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()

        self.ds = hf_ds
        self.tokenize = tokenize
        self.max_words = max_words

    def tokenize_and_extract(self, item):
        """
        item -  {
                    "id": "1126066/66",
                    "text": ["place the chicken tenders on a small", ...],
                    "begin": [1490, ...],
                    "end": [1720, ...],
                    "feat_file": "./data/clip_226_1fps/1126066/66/front_compressed.npy",
                    "timespan": {'begin': 1490, 'end': 38700}
                }
        """

        # time tokens
        timespan = item["timespan"]["end"] - item["timespan"]["begin"]
        time_tokens = np.array(item["begin"] + item["end"], dtype=np.longfloat)
        time_tokens = np.floor(time_tokens * 100 / timespan).astype(np.int32)

        time_tokens = [f"<{i}>" for i in time_tokens]
        begin_tokens = time_tokens[: len(item["text"])]
        end_tokens = time_tokens[len(item["text"]) :]

        text = "".join([f"{b}{e}{t}" for t, b, e in zip(item["text"], begin_tokens, end_tokens)])

        # tokenization
        net_input = self.tokenize(text)

        # segment id
        net_input["token_type_ids"] = [0 for _ in range(len(net_input["input_ids"]))]

        # video extraction
        net_input["video"] = np.load(item["feat_file"]).astype(np.float16)
        net_input["video_mask"] = np.zeros(net_input["video"].shape[:-1])

        return net_input

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        return self.tokenize_and_extract(item)