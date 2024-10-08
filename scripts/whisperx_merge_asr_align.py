import os
import pickle
import sys
from tqdm import tqdm
import multiprocessing
import math

files = os.listdir(sys.argv[1])

def to_video_id(x):
    return '_'.join(x.split('/')[-3:-1])

def read(file):
    try:
        return file.split('.')[0], pickle.load(open(os.path.join(sys.argv[1], file), 'rb'))
    except:
        return None

bs = 100000
n_batches = math.ceil(len(files) / bs)
for i in tqdm(range(n_batches)):
    if os.path.exists(sys.argv[2][:-4] + '_' + str(i) + '.pkl'):
        continue
    res = {}
    with multiprocessing.Pool(24) as p:
        results = p.map(read, files[i * bs: (i + 1) * bs])

    for x in results:
        if x is None:
            continue
        res[x[0]] = x[1]

    print(f"Saving {sys.argv[2][:-4] + '_' + str(i) + '.pkl'}")
    pickle.dump(res, open(sys.argv[2][:-4] + '_' + str(i) + '.pkl', 'wb'))

    bis = {}
    for x in res:
        bis[x] = {'segments': [{y: seg[y] for y in ['start', 'end', 'text']} for seg in res[x]['segments']]}
        if 'language' in res[x]:
            bis[x]['language'] = res[x]['language']

    print(f"Saving {sys.argv[2][:-4] + '_' + str(i) + '_proc.pkl'}")
    pickle.dump(bis, open(sys.argv[2][:-4] + '_' + str(i) + '_proc.pkl', 'wb'))

res = {}
for i in tqdm(range(n_batches)):
    tmp = pickle.load(open(sys.argv[2][:-4] + '_' + str(i) + '_proc.pkl', 'rb'))
    for x in tmp:
        res[x] = tmp[x]
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
with multiprocessing.Pool(24) as p:
    results = p.map(process, list(res))

out = {}
for x in results:
    vid = x['video_id']
    del x['video_id']
    out[vid] = x

asr = pickle.load(open(sys.argv[3], 'rb'))
for x in asr:
    if x not in out:
        out[x] = asr[x]

print(f"Saving {sys.argv[2][:-4] + '_proc.pkl'}")
pickle.dump(out, open(sys.argv[2][:-4] + '_proc.pkl', 'wb'))
