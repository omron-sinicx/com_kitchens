import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask


class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
        self,
        json_path,
        features_path,
        max_feats=100,
        features_dim=768,
        tokenizer=None,
        subtitles_path=None,
        num_bins=100,
        max_input_tokens=1000,
        max_output_tokens=256,
        noise_density=0.25,
        mean_noise_span_length=5,
        recipes_path=None,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.tokenizer = tokenizer
        self.subs = None
        self.subs_path = None
        if subtitles_path and os.path.exists(subtitles_path) and os.path.isdir(subtitles_path):
            self.subs_path = subtitles_path
        elif subtitles_path and os.path.exists(subtitles_path):
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            print("No subtitles given or found.")
        self.num_bins = num_bins
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_text_tokens = len(tokenizer) - num_bins
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

        self.recipes_path = recipes_path

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = th.from_numpy(np.load(features_path)).float()

        sampled = []
        if len(video) > self.max_feats:
            for j in range(self.max_feats):
                # sampled.append(video[(j * len(video)) // self.max_feats])
                sampled.append((j * len(video)) // self.max_feats)
            video = th.stack([video[s] for s in sampled])
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
            sampled = list(range(len(video))) + [-1 for s in range(self.max_feats - video_len)]
        else:
            video_len = self.max_feats
            sampled = list(range(len(video)))

        return video, sampled

    def _get_recipe(self, video_id):
        recipe_path = os.path.join(self.recipes_path, video_id.replace("_","/"), "gold_recipe.json")

        if os.path.exists(recipe_path):
            return json.load(open(recipe_path))
        else:
            return None

    def _get_ynodes(self, recipe):
        aps = recipe['actions_by_person']

        def get_timespan(ynode):
            start = int(ynode["meta_info"]['before'][0]['frame'])
            end = int(ynode["meta_info"]['after'][0]['frame'])
            return start / 30., end / 30.
        
        def get_input_of(ynode):
            return ynode['is_input_of']
        
        def get_output_of(ynode):
            return ynode['is_output_of']

        def get_destinated_to(ynode):
            return ynode['is_destinated_to']

        ynode_dict = {}
        for x_id, xnode in aps.items():
            for y_id, ynode in xnode.items():
                ynode_info = {
                    'timespan': get_timespan(ynode),
                    "input_of": get_input_of(ynode),
                    "output_of": get_output_of(ynode),
                    "destinated_to": get_destinated_to(ynode)
                }
                ynode_dict[f"{x_id}-{y_id}"] = ynode_info

        return ynode_dict

    def _yid_to_samples_dict(self, ynodes, sample_secs):
        map_dict = {}
        for yid, ynode in ynodes.items():
            start, end = ynode['timespan']

            matched = []
            for sample_sec in sample_secs:
                if start <= sample_sec and sample_sec <= end:
                    matched.append(sample_sec)
            
            map_dict[yid] = matched
        
        return map_dict
    
    def _sample_to_yids_dict(self, ynodes, sample_secs):
        map_dict = {}
        for sample_sec in sample_secs:
            matched = []
            for yid, ynode in ynodes.items():
                start, end = ynode['timespan']
                if start <= sample_sec and sample_sec <= end:
                    matched.append(yid)
            map_dict[sample_sec] = matched
        
        return map_dict

    def _get_ap_transits(self, recipe, sample_secs):

        if recipe is None:
            return None
        
        ynodes = self._get_ynodes(recipe)

        sample2yids = self._sample_to_yids_dict(ynodes, sample_secs)
        yid2samples = self._yid_to_samples_dict(ynodes, sample_secs)

        ap_transits = th.zeros((len(sample_secs), len(sample_secs)))

        # determine current aps
        for i, sample_sec in enumerate(sample_secs):
            matched_yids = sample2yids[sample_sec]
            
            for yid in matched_yids:
                ynode = ynodes[yid]
                # input_of
                transit_ynodes = ynode["input_of"]
                transit_samples = sum([yid2samples[_] for _ in transit_ynodes], [])
                transit_offsets = [sample_secs.index(s) for s in transit_samples]
                ap_transits[i].index_fill_(0, th.LongTensor(transit_offsets), 1)
            
                # output_of
                transit_ynodes = ynode["output_of"]
                transit_samples = sum([yid2samples[_] for _ in transit_ynodes], [])
                transit_offsets = [sample_secs.index(s) for s in transit_samples]
                ap_transits[i].index_fill_(0, th.LongTensor(transit_offsets), 2)
                
                # output_of
                transit_ynodes = ynode["destinated_to"]
                transit_samples = sum([yid2samples[_] for _ in transit_ynodes], [])
                transit_offsets = [sample_secs.index(s) for s in transit_samples]
                ap_transits[i].index_fill_(0, th.LongTensor(transit_offsets), 3)
        
        return ap_transits

    def time_tokenize(self, x, duration, num_bins):
        time_token = int(float((num_bins - 1) * x) / float(duration))
        assert time_token <= self.num_bins
        return time_token + self.num_text_tokens

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]
        video, sample_ids = self._get_video(video_id)
        duration = annotations["duration"]

        if self.recipes_path is not None:
            recipe = self._get_recipe(video_id)
            ap_transits = self._get_ap_transits(recipe, sample_ids)
        else:
            ap_transits = None

        # get subtitles
        if (self.subs is not None and video_id in self.subs) or (self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id + '.pkl'))):
            if (self.subs is not None and video_id in self.subs):
                sub = self.subs[video_id]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id + '.pkl'), 'rb'))

            to_keep = [(x >= 0 and y <= duration) for x, y in zip(sub["start"], sub["end"])]
            if not any(to_keep):  # no subtitles
                input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()
            else:
                sub["start"] = [x for i, x in enumerate(sub["start"]) if to_keep[i]]
                sub["end"] = [x for i, x in enumerate(sub["end"]) if to_keep[i]]
                sub['text'] = [self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]]
                time_input_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                                    self.time_tokenize(ed, duration, self.num_bins)])
                                     for st, ed in zip(sub['start'], sub['end'])]
                text_input_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_input_tokens,
                                                    padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                                     for x in sub['text']]
                input_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]
                input_tokens = th.cat(input_tokens, 0)
                input_tokens = input_tokens[:self.max_input_tokens - 1]
                input_tokens = th.cat([input_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)
        else:
            input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()

        # denoising sequence
        if len(input_tokens) > 1:
            mask_indices = np.asarray(
                [random_spans_noise_mask(len(input_tokens), self.noise_density, self.mean_noise_span_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.tokenizer, self.num_bins)
            labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.tokenizer, self.num_bins)

            denoising_output_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), labels_sentinel, self.tokenizer)).squeeze(0)
            denoising_input_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), input_ids_sentinel, self.tokenizer)).squeeze(0)
        else:
            input_tokens = th.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = th.LongTensor([0])
            denoising_output_tokens = input_tokens

        # dvc/vcg sequence
        captions = [self._get_text(x) for x in annotations['sentences']]
        time_output_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                             self.time_tokenize(ed, duration, self.num_bins)])
                              for st, ed in annotations['timestamps']]
        text_output_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_output_tokens,
                                             padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                              for x in captions]
        output_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_output_tokens, text_output_tokens)]
        output_tokens = th.cat(output_tokens, 0)
        output_tokens = output_tokens[:self.max_output_tokens - 1]
        output_tokens = th.cat([output_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)

        item = {
            "video_id": video_id,
            "duration": duration,
            "video": video,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }

        if ap_transits is not None:
            item.update({
                "output_ap_transits": ap_transits
            })
        
        return item

def densevideocaptioning_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    input_tokens = [batch[i]["input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in input_tokens)
    for i in range(bs):
        if len(input_tokens[i]) < max_input_len:
            input_tokens[i] = th.cat([input_tokens[i], th.zeros(max_input_len - len(input_tokens[i])).long()], 0)
    input_tokens = th.stack(input_tokens)
    output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
    max_output_len = max(len(x) for x in output_tokens)
    for i in range(bs):
        if len(output_tokens[i]) < max_output_len:
            output_tokens[i] = th.cat([output_tokens[i], th.zeros(max_output_len - len(output_tokens[i])).long()], 0)
    output_tokens = th.stack(output_tokens)
    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in denoising_input_tokens)
    for i in range(bs):
        if len(denoising_input_tokens[i]) < max_input_len:
            denoising_input_tokens[i] = th.cat(
                [denoising_input_tokens[i], th.zeros(max_input_len - len(denoising_input_tokens[i])).long()], 0)
    denoising_input_tokens = th.stack(denoising_input_tokens)
    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(bs)]
    max_denoising_output_len = max(len(x) for x in denoising_output_tokens)
    for i in range(bs):
        if len(denoising_output_tokens[i]) < max_denoising_output_len:
            denoising_output_tokens[i] = th.cat([denoising_output_tokens[i], th.zeros(
                max_denoising_output_len - len(denoising_output_tokens[i])).long()], 0)
    denoising_output_tokens = th.stack(denoising_output_tokens)

    out = {
        "video_id": video_id,
        "duration": duration,
        "video": video,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
    }

    # ap_transits
    if 'output_ap_transits' in batch[0]:
        out.update({
            "output_ap_transits": th.stack([b["output_ap_transits"] for b in batch])
        })

    return out


def build_densevideocaptioning_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_val_json_path
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        features_path = args.vitt_features_path
        subtitles_path = args.vitt_subtitles_path
    elif dataset_name == "chapters":
        if split == "train":
            json_path = args.chapters_train_json_path
        elif split == "val":
            json_path = args.chapters_val_json_path
        elif split == "test":
            json_path = args.chapters_test_json_path
        else:
            raise NotImplementedError
        features_path = args.chapters_features_path
        subtitles_path = args.chapters_subtitles_path
    elif dataset_name == "comk":
        json_path = getattr(args, f"{split}_json_path")
        features_path = args.features_path
        subtitles_path = args.subtitles_path
    else:
        raise NotImplementedError
    return DenseVideoCaptioning_Dataset(json_path=json_path,
                                        features_path=features_path,
                                        max_feats=args.max_feats,
                                        features_dim=args.features_dim,
                                        tokenizer=tokenizer,
                                        subtitles_path=subtitles_path,
                                        num_bins=args.num_bins,
                                        max_input_tokens=args.max_input_tokens,
                                        max_output_tokens=args.max_output_tokens,
                                        recipes_path=args.recipes_path,
                                        )
