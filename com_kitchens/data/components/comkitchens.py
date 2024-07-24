import json
import os

import spacy

from com_kitchens import utils

log = utils.get_pylogger(__name__)
nlp = spacy.load("en_core_web_sm")


def iter_ap_boundaries(ap_ids, ap_segment="end"):
    assert ap_segment in ("begin", "end")

    last_ap_id, last_ap_idx = None, None
    for i, ap_id in enumerate(ap_ids):
        # at the ap index
        if ap_id:
            if last_ap_id:
                yield last_ap_id, (last_ap_idx + 1 if ap_segment == "end" else i)

            last_ap_id = ap_id
            last_ap_idx = i

    else:
        yield last_ap_id, len(ap_ids)


def iter_ap_spans(ap_ids, ap_segment="end"):
    ap_bounds = list(iter_ap_boundaries(ap_ids, ap_segment))

    ap_ids = [i[0] for i in ap_bounds]
    ap_ends = [i[1] for i in ap_bounds]
    ap_spans = list(zip(ap_ids, list(zip([0] + ap_ends, ap_ends))))

    return ap_spans


def segment_by_ap(recipe, cumulative=False, ap_segment="end", sep=""):
    captions_by_ap = {}

    def get_fist_y_node(ap_id):
        return list(recipe["actions_by_person"][ap_id].values())[0]

    def get_last_y_node(ap_id):
        return list(recipe["actions_by_person"][ap_id].values())[-1]

    def get_end_frame(ap_id):
        meta_info = get_last_y_node(ap_id)["meta_info"]
        assert "before" in meta_info and "after" in meta_info
        assert len(meta_info["after"]) == 1

        return int(meta_info["after"][0]["frame"])

    if cumulative:
        # to avoid use `global` statement, store elements in a list
        before_frames = []

        def get_begin_frame(ap_id):
            meta_info = get_fist_y_node(ap_id)["meta_info"]
            before_frame = int(meta_info["before"][0]["frame"])
            before_frames.append(before_frame)
            return before_frames[0]

        cum_words = []

        def get_words(ap_words):
            cum_words.extend(ap_words)
            return cum_words

    else:

        def get_begin_frame(ap_id):
            meta_info = get_fist_y_node(ap_id)["meta_info"]
            return int(meta_info["before"][0]["frame"])

        def get_words(ap_words):
            return ap_words

    for step in recipe["steps"]:
        step_words = step["words"]
        for ap_id, ap_span in iter_ap_spans(step["ap_ids"], ap_segment=ap_segment):
            # begin and end frame
            begin_frame = get_begin_frame(ap_id)
            end_frame = get_end_frame(ap_id)

            # words
            ap_words = step_words[ap_span[0] : ap_span[1]]
            words = get_words(ap_words)

            # extract only noun and verbs
            words = _extract_noun_verbs(words)

            captions_by_ap[ap_id] = {
                "recipe": sep.join(words),
                "begin": begin_frame,
                "end": end_frame,
            }

    return captions_by_ap


def _extract_noun_verbs(words):
    noun_verbs = []
    sentence = " ".join(words)
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ in ["NOUN", "VERB"]:
            noun_verbs.append(token.text)
    return noun_verbs


def _ap_dict_to_list(xnode: dict):
    def ap_node_to_dict(node):
        y, props = node
        props["y"] = y
        return props

    return [
        {"x": x, "nodes": [ap_node_to_dict(node) for node in ynodes.items()]}
        for x, ynodes in xnode.items()
    ]


def _get_video_timespan(recipe):
    aps = _ap_dict_to_list(recipe["actions_by_person"])

    begin_frame = aps[0]["nodes"][0]["meta_info"]["before"][0]["frame"]
    end_frame = aps[-1]["nodes"][0]["meta_info"]["after"][-1]["frame"]

    return int(begin_frame), int(end_frame)


def _load_recipes(
    dat_path, recipe_dir, video_dir, video_ids=None, lang=None, ignore_absent=False, **kwargs
):
    def load_recipe_json(vid):
        recipe_path = os.path.join(recipe_dir, vid, "gold_recipe.json")
        if not os.path.exists(recipe_path):
            return None

        recipe_json = json.load(open(recipe_path))
        recipe_json["path"] = recipe_path

        if lang:
            trans_path = os.path.join(recipe_dir, vid, f"gold_recipe_translation_{lang}.json")
            if not os.path.exists(trans_path):
                return None

            trans_json = json.load(open(trans_path))
            assert isinstance(trans_json, dict)

            for key, value in trans_json.items():
                recipe_json[key] = value

        return recipe_json

    if video_ids is None:
        video_ids = [line.strip() for line in open(dat_path, encoding="utf-8")]

    key = 0
    for vid in video_ids:
        recipe_data = load_recipe_json(vid)

        if recipe_data is None:
            if ignore_absent:
                continue
            else:
                raise RuntimeError(f"Failed to load data: {vid}")

        yield key, recipe_data
        key += 1


def get_raw_examples(
    dat_path, recipe_dir, video_dir, video_ids=None, lang=None, ignore_absent=False, **kwargs
):
    for key, recipe_data in _load_recipes(
        dat_path, recipe_dir, video_dir, video_ids, lang, ignore_absent
    ):
        recipe_data["actions_by_person"] = _ap_dict_to_list(recipe_data["actions_by_person"])
        yield key, recipe_data


def get_retrieval_examples(
    dat_path,
    recipe_dir,
    video_dir,
    lang=None,
    ap_segment="end",
    ignore_absent=False,
    sep="",
    segments_by_frame=False,
    frame_ratios=[0.25, 0.5, 0.75, 1.0],
    cumulative=True,
    # map string id to int id to avoid tokenization
    recipe_id_dict={},
    kitchen_id_dict={},
    **kwargs,
):
    video_ids = [line.strip() for line in open(dat_path, encoding="utf-8")]

    key = 0
    for vid, (_, recipe) in zip(
        video_ids,
        _load_recipes(dat_path, recipe_dir, video_dir, lang=lang, ignore_absent=ignore_absent),
    ):
        recipe_id, kitchen_id = (s for s in vid.split("/"))
        captions_by_ap = segment_by_ap(recipe, cumulative=cumulative, ap_segment=ap_segment, sep=sep)

        if recipe_id not in recipe_id_dict:
            recipe_id_dict[recipe_id] = len(recipe_id_dict)
        if kitchen_id not in kitchen_id_dict:
            kitchen_id_dict[kitchen_id] = len(kitchen_id_dict)
        
        recipe_id = recipe_id_dict[recipe_id]
        kitchen_id = kitchen_id_dict[kitchen_id]

        if segments_by_frame:
            aps2 = segment_by_ap(recipe, cumulative=False, ap_segment=ap_segment, sep=sep)

            aps2 = sorted(aps2.items(), key=lambda ap: ap[1]["begin"])

            def find_nearest_ap(frame):
                last_ap_id = -1
                for ap_id, ap_seg in aps2:
                    if frame >= ap_seg["begin"]:
                        last_ap_id = ap_id
                    else:
                        return last_ap_id
                else:
                    if last_ap_id != -1:
                        return last_ap_id
                    else:
                        raise RuntimeError("Cannot find a nearest AP")

            begin, end = _get_video_timespan(recipe)
            duration = end - begin

            for frame_ratio in frame_ratios:
                frame = int(frame_ratio * duration)
                ap_id = find_nearest_ap(frame)
                ap_seg = captions_by_ap[ap_id] if ap_id != -1 else None
                if not ap_seg:
                    log.warn(f"Cannot find AP segmentation for the frame of {frame} in the video {vid}")

                frame_dir = os.path.join(video_dir, vid)

                if not os.path.exists(frame_dir):
                    err_msg = f"Frame directory not found: {frame_dir}"
                    if ignore_absent or "99999999" in frame_dir:
                        # log.warn(err_msg)
                        # print("Skipping Extra Resource: ", frame_dir)
                        continue
                    else:
                        raise RuntimeError(err_msg)  # ここでframe_dirが見つからなくてエラー

                yield key, {
                    "recipe_id": recipe_id,
                    "kitchen_id": kitchen_id,
                    "ap_id": int(ap_id),
                    "text": ap_seg["recipe"] if ap_seg else "",
                    "begin": 1,
                    "end": frame,
                    "frame_dir": os.path.join(video_dir, vid),
                    "is_query": True,
                    "is_pool": False,
                }
                key += 1

        for ap_id, ap_seg in captions_by_ap.items():
            frame_dir = os.path.join(video_dir, vid)

            if "99999999" in frame_dir:
                # Skip checking if frame dir points to dummy
                # log.warn(f"Skipping CHECKING Extra Resource: {frame_dir}")
                pass
            elif not os.path.exists(frame_dir):
                # Stop if frame dir does not exist
                err_msg = f"Frame directory not found: {frame_dir}"
                if ignore_absent:
                    log.warn(err_msg)
                    continue
                else:
                    raise RuntimeError(err_msg)

            yield key, {
                "recipe_id": recipe_id,
                "kitchen_id": kitchen_id,
                "ap_id": ap_id,
                "text": ap_seg["recipe"],
                "begin": ap_seg["begin"],
                "end": ap_seg["end"],
                "frame_dir": os.path.join(video_dir, vid),
                "is_query": not segments_by_frame,
                "is_pool": True,
            }
            key += 1


def get_dvc_examples(
    dat_path,
    recipe_dir,
    video_dir,
    feat_name="front_compressed.npy",
    lang=None,
    ap_segment="end",
    ignore_absent=False,
    sep="",
    **kwargs,
):
    video_ids = [line.strip() for line in open(dat_path, encoding="utf-8")]

    key = 0
    for vid, (_, recipe) in zip(
        video_ids,
        _load_recipes(
            dat_path, recipe_dir, video_dir, video_ids, lang=lang, ignore_absent=ignore_absent
        ),
    ):
        captions_by_ap = segment_by_ap(recipe, cumulative=False, ap_segment=ap_segment, sep=sep)

        begins = [v["begin"] for v in captions_by_ap.values()]
        ends = [v["end"] for v in captions_by_ap.values()]

        feat_file = os.path.join(
            video_dir, str(recipe["recipe_id"]), str(recipe["kitchen_id"]), feat_name
        )

        if not os.path.exists(feat_file):
            err_msg = f"Feature file not found: {feat_file}"
            if ignore_absent:
                log.warn(err_msg)
                continue
            else:
                raise RuntimeError(err_msg)

        yield key, {
            "id": vid,
            "ap_id": [k for k in captions_by_ap.keys()],
            "text": [v["recipe"] for v in captions_by_ap.values()],
            "begin": begins,
            "end": ends,
            "timespan": {"begin": min(begins), "end": max(ends)},
            "feat_file": feat_file,
        }
        key += 1
