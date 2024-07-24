import json
import os
import sys

from com_kitchens import utils

log = utils.get_pylogger(__name__)


def transform_recipe_json(input_root, output_root):
    recipe_json = {}
    recipe_paths = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file == "gold_recipe_translation_en.json":
                recipe_paths.append(root)

    for recipe_path in recipe_paths:
        recipe_path_en = os.path.join(recipe_path, "gold_recipe_translation_en.json")
        recipe_path_ja = os.path.join(recipe_path, "gold_recipe.json")

        recipe_id = "/".join(recipe_path.split("/")[-2:])
        recipe_json[recipe_id] = {"captions": {}}
        captions = recipe_json[recipe_id]["captions"]

        # get captions on each steps of recipes from json file.
        with open(recipe_path_en) as f:
            recipe_en = json.load(f)
            caption_by_ap = []
            for step in recipe_en["steps"]:
                # stack words on each steps of recipes by ap_id.
                recipe_ap_state = None
                for i, (ap_id, caption_word) in enumerate(zip(step["ap_ids"], step["words"])):
                    if ap_id is not None:
                        if recipe_ap_state is not None:
                            captions[recipe_ap_state] = {}
                            captions[recipe_ap_state]["recipe"] = " ".join(caption_by_ap)
                        recipe_ap_state = ap_id
                    # if this is the last word of the last step, add caption to caption_by_ap_dict.
                    if i == len(step["ap_ids"]) - 1:
                        captions[recipe_ap_state] = {}
                        captions[recipe_ap_state]["recipe"] = " ".join(caption_by_ap)
                    # lower-casing
                    caption_by_ap.append(caption_word.lower())

        # get key frames by ap_id from json file.
        begin_frame = sys.maxsize
        end_frame = -1
        with open(recipe_path_ja) as f:
            recipe_ja = json.load(f)
            # key frames are stored in recipe_ja["actions_by_person"][ap_id]["meta_info"][["before"|"after"]]
            # use only "after", we set before "000000" because we use the video from the beginning without any edit.
            for ap_id in list(captions.keys()):
                ap = recipe_ja["actions_by_person"][ap_id]
                subap_id, subap = list(ap.items())[0]

                # assert len(ap) == 1
                meta_info = subap["meta_info"]

                assert "before" in meta_info and "after" in meta_info
                assert len(meta_info["before"]) in (0, 1)
                assert len(meta_info["after"]) == 1

                before_frame = (
                    None if len(meta_info["before"]) == 0 else int(meta_info["before"][0]["frame"])
                )
                after_frame = int(meta_info["after"][0]["frame"])

                if before_frame is not None and before_frame < begin_frame:
                    begin_frame = before_frame
                if after_frame > end_frame:
                    end_frame = after_frame

                # recipe_json[recipe_id][ap_id]["before"] = before_frame
                captions[f"{ap_id}-{subap_id}"] = {
                    "recipe": captions[ap_id]['recipe'],
                    "after": after_frame
                }
                del captions[ap_id]

            recipe_json[recipe_id]["timespan"] = {"begin": begin_frame, "end": end_frame}

    os.makedirs(output_root, exist_ok=True)
    json.dump(recipe_json, open(os.path.join(output_root, "captions.json"), "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Transform gold_recipe.json and gold_recipe_translation_en.json."
    )
    parser.add_argument("-i", "--input_root", type=str, default="data/main", 
                        help="input root")
    parser.add_argument("-o", "--output_root", type=str, default="data/ap",
                        help="output root")

    args = parser.parse_args()
    log.info(args)

    transform_recipe_json(args.input_root, args.output_root)
