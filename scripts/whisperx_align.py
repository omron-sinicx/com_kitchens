import whisperx
import argparse
import pandas as pd
import os
from tqdm import tqdm
import torch
import pickle

parser = argparse.ArgumentParser(description='Easy ASR extractor')
parser.add_argument('--csv', type=str, required=True,
                    help='input csv with video input path')
parser.add_argument('--asr', type=str, required=True,
                    help='path to extracted ASR w/ Whisper')
parser.add_argument('--output_path', type=str, required=True,
                    help='path where to save results')
parser.add_argument('--device', type=str, default='cuda',
                    help='device')
parser.add_argument('--language', type=str, default='en',
                    help="ASR language")
parser.add_argument('--model-dir', type=str, default='cache',
                    help="the directory to save models")

args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df.sample(frac=1)

def to_video_id(x):
    return '_'.join(x.split('/')[-3:-1])

mapping = {to_video_id(x): x for x in df['video_path']}

asr = pickle.load(open(args.asr, 'rb'))
videos = [(x, mapping[x]) for x in asr if asr[x].get('language', '') == args.language]
print(f"Processing {len(videos)} out of {len(asr)} videos")

with torch.no_grad():
    model_a, metadata = whisperx.load_align_model(language_code=args.language, device=args.device, model_dir=args.model_dir)

    for x in tqdm(videos):
        target_path = os.path.join(args.output_path, to_video_id(x[1]) + '.pkl')
        if os.path.exists(target_path):
            continue

        audio = whisperx.load_audio(x[1])

        try:
            aligned_asr = whisperx.align(asr[x[0]]["segments"], model_a, metadata, audio, args.device)
        except Exception as e:
            print(x, e)
            break
        pickle.dump(aligned_asr, open(target_path, 'wb'))
