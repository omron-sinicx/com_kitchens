# The COM Kitchens dataset

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://opensource.org/licenses/MIT)

[![license](https://img.shields.io/badge/template-lightning_hydra_template-purple.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

## Table of Contents

- [Authors](#authors)
- [Citation](#citation)
- [Dataset Details](#dataset-details)
- [Quick Start](#quick-start)
- [Environment](#environment)
- [License](#license)

## Authors

Koki Maeda(3,1)\*, Tosho Hirasawa(4,1)\*, Atsushi Hashimoto(1), Jun Harashima(2), Leszek Rybicki(2), Yusuke Fukasawa(2), Yoshitaka Ushiku(1)

(1) OMRON SINIC X Corp.
(2) COOKPAD Inc.
(3) Tokyo Institute of Technology
(4) Tokyo Metropolitan University

\*: Equally Contribution.
This work is done for the internship at OMRON SINIC X.

## Citation

> \[!NOTE\]
> ```
> @InProceedings{comkitchens_eccv2024,
>    author    = {Koki Maeda and Tosho Hirasawa and Atsushi Hashimoto and Jun Harashima and Leszek Rybicki and Yusuke Fukasawa and Yoshitaka Ushiku},
>    title     = {COM Kitchens: An Unedited Overhead-view Video Dataset as a Vision-Language Benchmark},
>    booktitle = {Proceedings of the European Conference on Computer Vision},
>    year      = {2024},
>}
> ```

## Dataset Details

This COMKitchens dataset provides cooking videos annotated with a structured visual action graph.
The dataset currently has two benchmarks:

- Dense Video Captioning on unedited fixed-viewpoint videos (DVC-FV)
- Online Recipe Retrieval (OnRR)

We provide all the dataset for the benchmarks and attach `.dat` files which represent the train/validation/test split.

### File Structure

```sh
data
├─ ap                # captions for each action-by-person entry
├─ frames            # frames extracted from videos (split into train/valid/test)
├─ frozenbilm        # features by FrozenBiLM (used by vid2seq)
└─ main              # recipes annotated by human
    └─ {recipe_id}      # recipe id
        └─ {kitchen_id} # kitchen id
            ├─ cropped_images                  # cropped images of bounding boxes for visual action graph
            ├─ frames                          # annotated frames for AP of visual action graph
            ├─ front_compressed.mp4            # recorded video
            ├─ annotations.xml                 # annotations in xml file format
            ├─ gold_recipe_translation_en.json # recipe annotations
            ├─ gold_recipe.json                # rewritten recipe (in Japanese)
            ├─ graph.dot                       # visual action graph
            ├─ graph.dot.pdf                   # visualization of visual action graph
            └─ obj.names
    ├── ingredients.txt                # ingredients list in the COM Kitchens dataset
    ├── ingredients_translation_en.txt # translated ingredients list in the COM Kitchens dataset
    ├── train.txt                      # list of recipe id in the train split
    └── val.txt                        # list of recipe id in the validation split
```

### Important files

#### gold_recipe.json

`gold_recipe.json` provides the recipe information, to which the visual action graph is attached.

| key                 | value        | description                                                                       |
| ------------------- | ------------ | --------------------------------------------------------------------------------- |
| "recipe_id"         | `str`        | recipe id                                                                         |
| "kitchen_id"        | `int`        | kitchen id                                                                        |
| "ingredients"       | `List[str]`  | ingredients list (in Japanese)                                                    |
| "ingredient_images" | `List[str]`  | path of the images of each ingredient                                             |
| "steps"             | `List[Dict]` | annotations by step                                                               |
| "steps/memo"        | `str`        | recipe sentence                                                                   |
| "steps/words"       | `List[str]`  | recipe split word by word                                                         |
| "steps/ap_ids"      | `List[Dict]` | Correspondence between AP and words                                               |
| "actions_by_person" | `List[str]`  | annotation of the visual action graph, including the time span and bounding boxes |

#### {recipe_id}/{kitchen_id}/gold_recipe_translation_en.json

`gold_recipe_translation_en.json` provides only the translated recipe information.

| key            | value        | description                         |
| -------------- | ------------ | ----------------------------------- |
| "ingredients"  | `List[str]`  | ingredients list (in English)       |
| "steps"        | `List[Dict]` | annotations by step                 |
| "steps/memo"   | `str`        | recipe sentence                     |
| "steps/words"  | `List[str]`  | recipe split word by word           |
| "steps/ap_ids" | `List[Dict]` | Correspondence between AP and words |

### Download Procedure for COM Kitchens

> \[!NOTE\]
> [Application Form](https://www.nii.ac.jp/dsc/idr/rdata/COM_Kitchens/)
> English support will be available soon.

## Quick Start

### Dataset Preparation

1. Dataset Preparation
   1. Download annotation files and videos.
2. Preprocess
   1. Run `python -m com_kitchens.preprocess.video` for extracting all frames of the videos.
   2. Run `python -m com_kitchens.preprocess.recipe` for extracting all action-by-person entries of the videos.

> \[!WARNING\]
> While we extract all frames in preprocess for simplicity, you can save disk storage space by extracting only the frames you use with the annotation files.

### Online Recipe Retrieval (OnRR)

1. Training
   1. Run `sh scripts/onrr-train-xclip.sh` for simple start of trainings.
2. Evaluation
   1. Run `sh scripts/onrr-eval-xclip.sh {your/path/to/ckpt}` for the evaluation.

#### Training UniVL models in OnRR

For UniVL, we are required to extract s3d features of the videos.

1. Download `s3d_howto100m.pth` to `cache/s3d_howto100m.pth` or other path you configure.
2. Run `sh scripts/extract_s3d_features.sh` to extract s3d features.
3. Download pretrained model `univl.pretrained.bin` to `cache/univl.pretrained.bin` or other path you configure.
4. Then you can run `sh scripts/onrr-train-univl.sh` to train UniVL models.

### Dense Video Captioning on unedited fixed-viewpoint videos (DVC-FV)

1. Docker Images
   1. Run `make build-docker-images` to build docker images.
2. Preprocess
   1. Run `sh scripts/dvc-vid2seq-prep` to extract 
3. Training & Evaluation
   1. Run `sh scripts/vid2seq-zs.sh` to evaluate a pre-trained vid2seq model
   2. Run `sh scripts/vid2seq-ft.sh` to fine-tune and evaluate a vid2seq model
   3. RUn `sh scripts/vid2seq-ft-rl-as.sh` to fine-tune and evaluate a vid2seq model incorporating action graph as both relation labels and attention supervision (*RL+AS*)

## LICENSE

This project (other than the dataset) is licensed under the *MIT License*, see the LICENSE.txt file for details.