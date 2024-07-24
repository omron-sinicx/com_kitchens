import os
import argparse
from args import get_args_parser
from dvc import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    args.eval = True
    args.epochs = 0

    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)

    if not args.load:
        args.load = os.path.join(args.save_dir, "best_model.pth")

    main(args)
