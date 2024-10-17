import cv2
import os
import sys
import argparse
from .utils import load_anno_point, save_anno_point


def label_points(video_path, annotation_path):
    # Path to the video file
    if not os.path.isfile(video_path) or not video_path.endswith(".mp4"):
        print("Not a valid video path! Please provide a valid .mp4 file.")
        sys.exit(1)
    save_anno_name = video_path.replace(".mp4", "_point_anno.csv")

    # Check if annotation CSV exists
    if os.path.isfile(annotation_path) and annotation_path.endswith(".csv"):
        load_csv = True
    else:
        print("Not a valid csv file! Annotate from scratch.")
        load_csv = False

    # Prepare video capture
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Filter out invalid frames
    for frame_id in range(n_frames):
        print(f"Checking frame {n_frames - frame_id - 1}...")
        cap.set(1, n_frames - frame_id - 1)
        ret, image = cap.read()

        if not ret:
            continue
        else:
            n_frames = n_frames - frame_id
            break

    # Load or create new annotation data
    if load_csv:
        annotations = load_anno_point(annotation_path)
        if len(annotations) > n_frames:
            for i in range(n_frames, len(annotations)):
                del annotations[i]
            assert len(annotations) == n_frames
        elif len(annotations) < n_frames:
            for i in range(len(annotations), n_frames):
                annotations[i] = {
                    "Frame": i,
                    "Visibility": 0,
                    "X": -1,  # X coordinate for the point
                    "Y": -1,  # Y coordinate for the point
                }
        n_frames = len(annotations)
    else:
        annotations = {
            idx: {
                "Frame": idx,
                "Visibility": 0,
                "X": -1,  # X coordinate for the point
                "Y": -1,  # Y coordinate for the point
            }
            for idx in range(n_frames)
        }

    # Global variable for point drawing
    frame_no = 0

    # Mouse callback to draw the point
    def draw_point(event, x, y, flags, param):
        nonlocal image, frame_no, annotations
        if event == cv2.EVENT_LBUTTONDOWN:  # Capture the point on left click
            image[:] = original_image[:]
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(
                image,
                f"Frame: {frame_no}/{n_frames - 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Frame", image)
            annotations[frame_no] = {
                "Frame": frame_no,
                "X": x,
                "Y": y,
                "Visibility": 1,
            }

    # Main loop for video frames
    while True:
        cap.set(1, frame_no)  # Set the frame number to read
        ret, original_image = cap.read()
        if not ret:
            break

        image = original_image.copy()

        # Draw existing annotations for the current frame
        if annotations[frame_no]["Visibility"] == 1:
            cv2.circle(
                image,
                (annotations[frame_no]["X"], annotations[frame_no]["Y"]),
                5,
                (255, 0, 0),
                -1,
            )

        cv2.putText(
            image,
            f"Frame: {frame_no}/{n_frames - 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Frame", image)
        cv2.setMouseCallback("Frame", draw_point)

        # Display instructions
        print(f"Frame {frame_no}/{n_frames - 1}: Click to label a point.")
        # print(
        #     "Press\n's' to save the current frame, \n'n' for the next frame, \n'p' for the previous frame, \n'f' for the first frame, \n'l' for the last frame, \n'>' to jump forward 30 frames, \n'<' to jump back 30 frames, \n'q' to quit."
        # )
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):  # Save the current frame annotation
                save_anno_point(annotations, save_anno_name)
                break

            elif key == ord("x"):
                annotations[frame_no] = {
                    "Frame": frame_no,
                    "Visibility": 0,
                    "X": -1,
                    "Y": -1,
                }
                break

            elif key == ord("n"):  # Next frame
                frame_no += 1
                if frame_no >= n_frames:
                    print("Reached the end of the video, back to 1st frame.")
                    frame_no = 0
                break

            elif key == ord("p"):  # Previous frame
                frame_no -= 1
                if frame_no < 0:
                    print("Reached the first frame, back to last frame.")
                    frame_no = n_frames - 1
                break

            elif key == ord("f"):  # First frame
                frame_no = 0
                break

            elif key == ord("l"):  # Last frame
                frame_no = n_frames - 1
                break

            elif key == ord(">"):  # Jump forward 30 frames
                frame_no = min(n_frames - 1, frame_no + 30)
                break

            elif key == ord("<"):  # Jump backward 30 frames
                frame_no = max(0, frame_no - 30)
                break

            elif key == ord("q"):  # Quit
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vp",
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "-ap",
        "--annotation_path",
        type=str,
        required=False,
        default="",
        help="Path to the annotation file",
    )

    args = parser.parse_args()
    label_points(args.video_path, args.annotation_path)
