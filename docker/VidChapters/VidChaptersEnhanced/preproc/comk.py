from args import DATA_DIR
import json
import os
import subprocess

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

            captions_by_ap[ap_id] = {
                "recipe": sep.join(words),
                "begin": begin_frame,
                "end": end_frame,
            }

    return captions_by_ap

train_ids = set(open(os.path.join(DATA_DIR, 'main', 'train.dat')).read().split('\n')[:-1])
val_ids = set(open(os.path.join(DATA_DIR, 'main', 'val.dat')).read().split('\n')[:-1])

train, val = {}, {}
counter = {}

for vid in (train_ids | val_ids):
    video_id = vid.replace("/", "_")

    # video
    video_path = os.path.join(DATA_DIR, 'main', vid, "front_compressed.mp4")

    # duration
    result = subprocess.run(['ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout)

    # start/end of captions
    recipe_path = os.path.join(DATA_DIR, 'main', vid, "gold_recipe.json")
    recipe_en_path = os.path.join(DATA_DIR, 'main', vid, "gold_recipe_translation_en.json")

    recipe = json.load(open(recipe_path))
    recipe['steps'] = json.load(open(recipe_en_path))['steps']

    aps = segment_by_ap(recipe, ap_segment='begin', sep=' ')

    start = [ap['begin'] for ap in aps.values()]
    end = [ap['end'] for ap in aps.values()]

    out = {
        'duration': duration,
        'timestamps': [[st / 1000, ed / 1000] for st, ed in zip(start, end)],
        'sentences': [ap['recipe'] for ap in aps.values()],
        'path': video_id + '.npy'
    }

    if vid in train_ids:
        train[video_id] = out
    elif vid in val_ids:
        val[video_id] = out
    else:
        raise NotImplementedError

print("train/val = {}/{}".format(len(train), len(val)))
json.dump(train, open(os.path.join(DATA_DIR, 'vid2seq', 'train.json'), 'w'))
json.dump(val, open(os.path.join(DATA_DIR, 'vid2seq', 'val.json'), 'w'))
