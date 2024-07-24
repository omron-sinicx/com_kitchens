import os
import pickle
import sys
import multiprocessing

files = os.listdir(sys.argv[1])
res = {}

def read(file):
    return file.split('.')[0], pickle.load(open(os.path.join(sys.argv[1], file), 'rb'))

with multiprocessing.Pool() as p:
    results = p.map(read, files)

for x in results:
    res[x[0]] = x[1]

print(f'Saving {sys.argv[2]}')
pickle.dump(res, open(sys.argv[2], 'wb'))

def process(vid):
    texts, starts, ends = [], [], []
    for i in range(len(res[vid]['segments'])):
        text = res[vid]['segments'][i]['text']
        if text.strip():
            texts.append(text)
            starts.append(res[vid]['segments'][i]['start'])
            ends.append(res[vid]['segments'][i]['end'])
    return {'video_id': vid, 'text': texts, 'start': starts, 'end': ends}

print('Processing')
with multiprocessing.Pool() as p:
    results = p.map(process, list(res))

out = {}
for x in results:
    vid = x['video_id']
    del x['video_id']
    out[vid] = x

print(f"Saving {sys.argv[2][:-4] + '_proc.pkl'}")
pickle.dump(out, open(sys.argv[2][:-4] + '_proc.pkl', 'wb'))