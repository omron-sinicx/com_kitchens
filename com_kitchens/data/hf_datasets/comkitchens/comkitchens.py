# Lint as: python3
"""COM Kitchens: Cookpad OMron Kitchens dataset."""

import json
import os
from pathlib import Path

import datasets
from datasets import Value

from com_kitchens.data.components.comkitchens import (
    get_dvc_examples,
    get_raw_examples,
    get_retrieval_examples,
)

logger = datasets.logging.get_logger(__name__)

# _CITATION = """\
# @article{,
#        author = {},
#         title = "{}",
#       journal = {},
#          year = ,
#           eid = {},
#         pages = {},
# archivePrefix = {},
#        eprint = {},
# }
# """

_VERSION = "0.0.1"

_DESCRIPTION = """\
COM Kitchens: Cookpad OMron Kitchens dataset.
"""

# _URL = ""
# _URLS = {
#     "train": None,
#     "dev": None,
# }

_LANGS = ["en"]

_COMKITCHEN_RAW_FEATURES = datasets.Features(
    {
        "path": Value("string"),
        "recipe_id": Value("string"),
        "kitchen_id": Value("string"),
        "ingredients": [Value("string")],
        "ingredient_images": [Value("string")],
        "steps": [
            {
                "memo": Value("string"),
                "words": [Value("string")],
                "ap_ids": [Value("string")],
            }
        ],
        "actions_by_person": [
            {
                "x": Value("string"),
                "nodes": [
                    {
                        "y": Value("string"),
                        "image_paths": {
                            "before": [Value("string")],
                            "after": [Value("string")],
                            "dest": [Value("string")],
                        },
                        "meta_info": {
                            "before": [
                                {
                                    "frame": Value("string"),
                                    "bb": {
                                        "xtl": Value("float32"),
                                        "ytl": Value("float32"),
                                        "xbr": Value("float32"),
                                        "ybr": Value("float32"),
                                        "width": Value("int32"),
                                        "height": Value("int32"),
                                    },
                                }
                            ],
                            "after": [
                                {
                                    "frame": Value("string"),
                                    "bb": {
                                        "xtl": Value("float32"),
                                        "ytl": Value("float32"),
                                        "xbr": Value("float32"),
                                        "ybr": Value("float32"),
                                        "width": Value("int32"),
                                        "height": Value("int32"),
                                    },
                                }
                            ],
                            "dest": [
                                {
                                    "frame": Value("string"),
                                    "bb": {
                                        "xtl": Value("float32"),
                                        "ytl": Value("float32"),
                                        "xbr": Value("float32"),
                                        "ybr": Value("float32"),
                                        "width": Value("int32"),
                                        "height": Value("int32"),
                                    },
                                }
                            ],
                        },
                        "is_input_of": [Value("string")],
                        "is_output_of": [Value("string")],
                        "is_destinated_to": [Value("string")],
                        "word_index": Value("int64"),
                        "step_index": Value("int64"),
                    }
                ],
            }
        ],
    }
)

_COMKITCHEN_RETRIEVAL_FEATURES = datasets.Features(
    {
        "recipe_id": Value("int32"),  # {recipe_id}/{kitchen_id}/{ap_id}
        "kitchen_id": Value("int32"),
        "ap_id": Value("string"),
        "text": Value("string"),
        "begin": Value("int32"),
        "end": Value("int32"),
        "frame_dir": Value("string"),
        "is_query": Value("bool"),
        "is_pool": Value("bool"),
    }
)

_COMKITCHEN_DVC_FEATURES = datasets.Features(
    {
        "id": Value("string"),
        "ap_id": [Value("string")],
        "text": [Value("string")],
        "begin": [Value("int32")],
        "end": [Value("int32")],
        "feat_file": Value("string"),
        "timespan": {
            "begin": Value("int32"),
            "end": Value("int32"),
        },
    }
)

COMKITCHEN_DAT_NAMES = {
    datasets.Split.TRAIN: "train.dat",
    datasets.Split.VALIDATION: "val.dat",
    datasets.Split.TEST: "test.dat",
}


def set_absolute_path(obj):
    if isinstance(obj, str):
        return str(Path(obj).absolute())
    if isinstance(obj, dict):
        return {k: set_absolute_path(v) for k, v in obj.items()}
    else:
        return obj


def set_split_path(obj, split_names=None):
    if isinstance(obj, dict):
        return obj

    if os.path.isdir(obj) and isinstance(split_names, dict):
        return {k: os.path.join(obj, v) for k, v in split_names.items()}

    return {
        datasets.Split.TRAIN: obj,
        datasets.Split.VALIDATION: obj,
        datasets.Split.TEST: obj,
    }


class COMKitchensConfig(datasets.BuilderConfig):
    """BuilderConfig for COMKitchens."""

    def __init__(
        self,
        recipe_dir,
        video_dir,
        features,
        feature_proc,
        dat_files,
        lang,
        ap_segment,
        ignore_absent,
        sep,
        test_frame_ratios,
        cumulative,
        recipe_id_dict,
        kitchen_id_dict,
        **kwargs,
    ):
        """BuilderConfig for COMKitchens.
        Args:
          recipe_dir:
          video_dir: path of a directory contains video features (frames or extracted features)
          langs:
          features:
          feature_proc:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version(_VERSION), **kwargs)

        self.dat_files = set_absolute_path(dat_files)
        self.recipe_dir = set_absolute_path(recipe_dir)
        self.video_dir = set_absolute_path(video_dir)
        self.features = features
        self.feature_proc = feature_proc
        self.lang = lang
        self.ap_segment = ap_segment
        self.ignore_absent = ignore_absent
        self.sep = sep
        self.test_frame_ratios = test_frame_ratios
        self.cumulative = cumulative
        self.recipe_id_dict = recipe_id_dict
        self.kitchen_id_dict = kitchen_id_dict

class COMKitchens(datasets.GeneratorBasedBuilder):
    """COMKitchens: The Stanford Question Answering Dataset. Version 1.1."""

    DEFAULT_CONFIG_KWARGS = {
        "dat_files": set_split_path("./data/main", COMKITCHEN_DAT_NAMES),
        "recipe_dir": "./data/main",
        "lang": None,
        "ap_segment": "end",
        "ignore_absent": False,
        "sep": " ",
        "test_frame_ratios": [0.25, 0.5, 0.75, 1.0],
        "cumulative": True,
        "recipe_id_dict": {},
        "kitchen_id_dict": {},
    }

    BUILDER_CONFIG_CLASS = COMKitchensConfig
    BUILDER_CONFIGS = [
        COMKitchensConfig(
            name="raw",
            description="COM Kitchens dataset",
            **DEFAULT_CONFIG_KWARGS,
            video_dir="./data/main",
            features=_COMKITCHEN_RAW_FEATURES,
            feature_proc=get_raw_examples,
        ),
        COMKitchensConfig(
            name="retrieval",
            description="COM Kitchens dataset for recipe retrieval task",
            **DEFAULT_CONFIG_KWARGS,
            video_dir={
                datasets.Split.TRAIN: "./data/frames/train",
                datasets.Split.VALIDATION: "./data/frames/val",
                datasets.Split.TEST: "./data/frames/test",
            },
            features=_COMKITCHEN_RETRIEVAL_FEATURES,
            feature_proc=get_retrieval_examples,
        ),
        COMKitchensConfig(
            name="dvc",
            description="COM Kitchens dataset for dense video captioning task",
            **DEFAULT_CONFIG_KWARGS,
            video_dir="./data/clip_226_1fps",
            features=_COMKITCHEN_DVC_FEATURES,
            feature_proc=get_dvc_examples,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            supervised_keys=None,
            homepage="",
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # downloaded_files = dl_manager.download_and_extract(_URLS)

        def get_split_attr(attr, split):
            if isinstance(attr, str):
                return attr
            else:
                # `omegaconf.dictconfig.DictConfig` requires class-level matching
                # cast to dict to avoid KeyValidationError
                return dict(attr)[split]

        def get_split_gen_kwargs(split):
            return {
                "dat_path": get_split_attr(self.config.dat_files, split),
                "recipe_dir": get_split_attr(self.config.recipe_dir, split),
                "video_dir": get_split_attr(self.config.video_dir, split),
                "feature_proc": self.config.feature_proc,
                "lang": self.config.lang,
                "ap_segment": self.config.ap_segment,
                "ignore_absent": self.config.ignore_absent,
                "sep": self.config.sep,
                "segments_by_frame": split == datasets.Split.TEST,
                "frame_ratios": self.config.test_frame_ratios,
                "cumulative": self.config.cumulative,
            }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=get_split_gen_kwargs(datasets.Split.TRAIN),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=get_split_gen_kwargs(datasets.Split.VALIDATION),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=get_split_gen_kwargs(datasets.Split.TEST),
            ),
        ]

    def _generate_examples(
        self,
        dat_path,
        recipe_dir,
        video_dir,
        feature_proc,
        lang,
        ap_segment,
        ignore_absent,
        sep,
        segments_by_frame,
        frame_ratios,
        cumulative=True,
    ):
        """This function returns the examples in the raw (text) form.

        As this function is called along with the dataset generation, better to avoid includi any
        heavy operation.
        """
        logger.info("generating examples from %s", dat_path)

        yield from feature_proc(
            dat_path=dat_path,
            recipe_dir=recipe_dir,
            video_dir=video_dir,
            lang=lang,
            ap_segment=ap_segment,
            ignore_absent=ignore_absent,
            sep=sep,
            segments_by_frame=segments_by_frame,
            frame_ratios=frame_ratios,
            cumulative=cumulative,
        )


COMKITCHENS_TASKS = [cfg.name for cfg in COMKitchens.BUILDER_CONFIGS]

if __name__ == "__main__":
    from datasets import load_dataset

    kwargs = {
        "ap_segment": "begin",
        "ignore_absent": False,
        "sep": " ",
    }

    raw_data = load_dataset(__file__, "raw", **kwargs)
    print(raw_data)

    retrieval_data = load_dataset(__file__, "retrieval", **kwargs)
    print(retrieval_data)

    dvc_data = load_dataset(__file__, "dvc", **kwargs)
    print(dvc_data)
