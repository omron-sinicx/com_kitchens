import whisperx
import argparse
import pandas as pd
import os, sys
from tqdm import tqdm
import torch
import pickle

parser = argparse.ArgumentParser(description='Easy ASR extractor')
parser.add_argument('--csv', type=str,
                    help='input csv with video input path')
parser.add_argument('--type', type=str, default='large-v2', choices=['large-v2'],
                    help='model type')
parser.add_argument('--device', type=str, default='cuda',
                    help='device')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--language', type=str, default='en',
                    help="ASR language")
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df.sample(frac=1)

model = whisperx.load_model(args.type, device=args.device, language=args.language, asr_options={
    "repetition_penalty": 1, 
    "prompt_reset_on_temperature": 0.5,
    "no_repeat_ngram_size": 2,
})
if model.tokenizer is not None:
    print(f'Loaded {args.type} model for {model.tokenizer.language_code}.')

print("Starting extraction")
with torch.no_grad():
    for index, row in tqdm(df.iterrows()):
        video_path = row["video_path"]
        target_path = row["feature_path"]

        if os.path.exists(target_path):
            continue
        
        audio = whisperx.load_audio(video_path)
        try:
            result = model.transcribe(audio, batch_size=args.batch_size)
            if result['language'] != args.language:
                print(f"Bad detected language ({result['language']}) for {video_path}")
                
        except RuntimeError as e:
            print(video_path, e, file=sys.stderr)
            continue
        
        pickle.dump(result, open(target_path, 'wb'))
