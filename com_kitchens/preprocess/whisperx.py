#!/usr/bin/env python3
import pandas as pd
from glob import glob

def get_mapping(
        video_root: str = 'data/main',
        feature_root: str = 'data/frozenbilm',
    ):

    def to_feature_path(video_path):
        recipe_id, kitchen_id = video_path.split('/')[-3:-1]

        return f"{feature_root}/{recipe_id}_{kitchen_id}.pkl"

    data = pd.DataFrame({
        "video_path": list(glob(f'{video_root}/*/*/front_compressed.mp4'))
    })
    data["feature_path"] = data["video_path"].map(to_feature_path)

    return data

if __name__ == "__main__":
    import sys
    from tap import Tap
    class Args(Tap):
        video_root: str = 'data/main'
        feature_root: str = 'data/whisperx'
        csv_path: str = 'data/vid2seq/whisperx.csv'

    args = Args().parse_args()
    print(args, file=sys.stderr)

    data = get_mapping(
        video_root=args.video_root,
        feature_root=args.feature_root
    )

    data.to_csv(args.csv_path, index=False)
