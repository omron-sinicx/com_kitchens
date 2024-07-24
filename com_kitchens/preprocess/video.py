import os
import subprocess
import ffmpeg
from com_kitchens import utils

log = utils.get_pylogger(__name__)

def extract_video_frames(video_path, image_dir):
    """Extract all frames in a video and store to a directory per each minutes."""

    log.debug(f"Video: {video_path}")
    log.debug(f"Image directory: {image_dir}")

    os.makedirs(image_dir, exist_ok=True)

    probe = ffmpeg.probe(video_path)
    video_length_seconds = int(probe["format"]["duration"].split(".")[0])

    # 1分毎にサブディレクトリを作成して、その中にフレームを格納する
    for i in range(video_length_seconds // 60 + 1):
        os.makedirs(os.path.join(image_dir, str(i)), exist_ok=True)

        command = [
            "ffmpeg",
            "-ss",
            str(i * 60),
            "-t",
            "60",
            "-i",
            video_path,
            "-vframes",
            "1800",
            "-vcodec",  # video codec
            "mjpeg",
            f"{os.path.join(image_dir, str(i))}/frames_%05d.jpg",
        ]

        ffmpeg_process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        out, err = ffmpeg_process.communicate()
        retcode = ffmpeg_process.poll()


if __name__ == "__main__":
    import argparse
    import glob
    import logging
    from multiprocessing import Pool
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Picking up frames from videos for dataset making"
    )
    parser.add_argument("-i", "--input_root", type=str, default='data/main',
                        help="input root")
    parser.add_argument("-o", "--output_root", type=str, default='data/frames',
                        help="output root")
    parser.add_argument(
        "--cpu", type=int, default=None, required=False, help="number of worker process"
    )

    args = parser.parse_args()
    log.info(args)

    # Prepare (input video, output dir) pairs
    input_root = args.input_root
    output_root = args.output_root

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    def load_dat_file(dat_path):
        if os.path.exists(dat_path):
            return [x.strip("\n") for x in open(dat_path)]
        else:
            return []

    def prepare_input_output_pairs(input_root, output_root):
        input_videos = []
        output_image_dirs = []

        videos = {
            'train': load_dat_file(os.path.join(input_root, "train.dat")),
            'val': load_dat_file(os.path.join(input_root, "val.dat")),
            'test': load_dat_file(os.path.join(input_root, "test.dat"))
        }

        def get_split(video_id):
            for split, split_videos in videos.items():
                if video_id in split_videos:
                    return split
            else:
                return None

        # train, val, testのどれかは[train|val|test]_full.datに対応するものが書かれている
        for video in glob.glob(f"{input_root}/**/*.mp4", recursive=True):
            id = "/".join(video.split("/")[-3:-1])

            split = get_split(id)
            assert split is not None, f"id: {id} is not in train, val, test"

            input_videos.append(video)
            output_image_dirs.append(os.path.join(output_root, split, id))

        return input_videos, output_image_dirs

    input_videos, output_image_dirs = prepare_input_output_pairs(
        input_root=input_root, output_root=output_root
    )

    log.info(f"Processing {len(input_videos)} videos")

    # Extract frames on multiple CPUs
    with Pool(processes=args.cpu) as pool:
        log.info(f"Begin with {pool._processes} logical processors.")
        pool.starmap(
            extract_video_frames,
            [
                (input_video, output_image_dir)
                for input_video, output_image_dir in zip(input_videos, output_image_dirs)
            ],
        )
