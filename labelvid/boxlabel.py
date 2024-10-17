import cv2
import os
import sys
from .utils import load_anno_box, save_anno_box


def label_boxes(video_path, annotation_path):
    # Check if video file is valid
    if not os.path.isfile(video_path) or not video_path.endswith(".mp4"):
        print("Not a valid video path! Please modify path.")
        sys.exit(1)

    save_anno_name = video_path.replace(".mp4", "_box_anno.csv")

    # Load annotations if available
    if os.path.isfile(annotation_path) and annotation_path.endswith(".csv"):
        load_csv = True
    else:
        print("Not a valid csv file! Annotating from scratch.")
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
            continue  # Skip invalid frames
        else:
            n_frames = n_frames - frame_id  # Adjust total frame count
            break  # Stop after a valid frame is found

    # Load or create annotations
    if load_csv:
        annotations = load_anno_box(annotation_path)
        if len(annotations) > n_frames:
            annotations = annotations[:n_frames]
        elif len(annotations) < n_frames:
            for i in range(len(annotations), n_frames):
                annotations[i] = {
                    "Frame": i,
                    "Visibility": 0,
                    "X1": -1,
                    "Y1": -1,
                    "X2": -1,
                    "Y2": -1,
                }
    else:
        annotations = {
            idx: {
                "Frame": idx,
                "Visibility": 0,
                "X1": -1,  # Top-left corner of bounding box
                "Y1": -1,
                "X2": -1,  # Bottom-right corner of bounding box
                "Y2": -1,
            }
            for idx in range(n_frames)
        }

    # Global variables for box drawing
    drawing = False
    start_point = (-1, -1)
    end_point = (-1, -1)
    frame_no = 0

    # Mouse callback to draw the bounding box
    def draw_box(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, image, frame_no, annotations

        if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing the box
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:  # Update the box while dragging
            if drawing:
                end_point = (x, y)
                img_copy = image.copy()
                cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(
                    img_copy,
                    f"Frame: {frame_no}/{n_frames - 1}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Frame", img_copy)

        elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing the box
            drawing = False
            end_point = (x, y)
            image[:] = original_image[:]
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
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

            # Save the box coordinates to annotations
            annotations[frame_no] = {
                "Frame": frame_no,
                "X1": start_point[0],
                "Y1": start_point[1],
                "X2": end_point[0],
                "Y2": end_point[1],
                "Visibility": 1,
            }

    # Main loop for video frames
    while True:
        cap.set(1, frame_no)
        ret, original_image = cap.read()
        if not ret:
            break

        image = original_image.copy()

        if annotations[frame_no]["Visibility"] == 1:
            cv2.rectangle(
                image,
                (annotations[frame_no]["X1"], annotations[frame_no]["Y1"]),
                (annotations[frame_no]["X2"], annotations[frame_no]["Y2"]),
                (0, 255, 0),
                2,
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
        cv2.setMouseCallback("Frame", draw_box)

        print(f"Frame {frame_no}/{n_frames - 1}: Draw a box with the mouse.")
        print("Press 's' to save, 'n' for next, 'p' for previous, 'q' to quit.")
        print(
            "Press 'f' for first frame, 'l' for last frame, '>' to jump 30 frames forward, '<' to jump back 30 frames."
        )

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):  # Save annotations
                save_anno_box(annotations, save_anno_name)
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

            elif key == ord("x"):
                annotations[frame_no] = {
                    "Frame": frame_no,
                    "Visibility": 0,
                    "X1": -1,
                    "Y1": -1,
                    "X2": -1,
                    "Y2": -1,
                }
                break

            elif key == ord("q"):  # Quit
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        drawing = False
        start_point = (-1, -1)
        end_point = (-1, -1)

    cap.release()
    cv2.destroyAllWindows()


# If this file is run directly, parse arguments and start box labeling
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vp", "--video_path", type=str, required=True, help="Path to the video file"
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
    label_boxes(args.video_path, args.annotation_path)
