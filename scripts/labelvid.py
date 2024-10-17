import argparse
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from labelvid.boxlabel import label_boxes
from labelvid.pointlabel import label_points


def main():
    parser = argparse.ArgumentParser(description="Label video points or boxes")
    subparsers = parser.add_subparsers(dest="command")

    # Box label subcommand
    box_parser = subparsers.add_parser("boxlabel")
    box_parser.add_argument(
        "-vp", "--video_path", type=str, required=True, help="Path to video file"
    )
    box_parser.add_argument(
        "-ap",
        "--annotation_path",
        type=str,
        default="",
        required=False,
        help="Path to annotation file",
    )

    # Point label subcommand
    point_parser = subparsers.add_parser("pointlabel")
    point_parser.add_argument(
        "-vp", "--video_path", type=str, required=True, help="Path to video file"
    )
    point_parser.add_argument(
        "-ap",
        "--annotation_path",
        type=str,
        default="",
        required=False,
        help="Path to annotation file",
    )

    args = parser.parse_args()

    if args.command == "boxlabel":
        label_boxes(args.video_path, args.annotation_path)
    elif args.command == "pointlabel":
        label_points(args.video_path, args.annotation_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
