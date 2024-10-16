import cv2
import os
import sys
import csv
import argparse
from utils import load_anno_box, save_anno_box

parser = argparse.ArgumentParser()
parser.add_argument(
    "-vp",
    "--video_path",
    type=str,
    default="data/badminton_demo.mp4",
    help="Path to the video file",
)
parser.add_argument(
    "-ap", "--annotation_path", type=str, default="", help="Path to the annotation file"
)

args = parser.parse_args()

# Path to the video file
video_path = args.video_path  # Change this to your video path
if not os.path.isfile(video_path) or not video_path.endswith(".mp4"):
    print("Not a valid video path! Please modify path in parser.py --label_video_path")
    sys.exit(1)
save_anno_name = video_path.replace(".mp4", "_anno.csv")

anno_path = args.annotation_path
if os.path.isfile(anno_path) and anno_path.endswith(".csv"):
    load_csv = True
else:
    print("Not a valid csv file! Annotate from scratch.")
    load_csv = False


# Prepare video capture
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# filter out invalid frames
for frame_id in range(n_frames):
    print(f"Checking frame {n_frames - frame_id - 1}...")
    cap.set(1, n_frames - frame_id - 1)
    ret, image = cap.read()

    # Check if the frame was successfully grabbed
    if not ret:
        continue  # Skip to the next iteration if frame is not valid
    else:
        n_frames = (
            n_frames - frame_id
        )  # Adjust total frame count if a valid frame is found
        break  # Exit the loop since a valid frame was found

# load offered csv or create a new one
if load_csv:
    annotations = load_anno_box(anno_path)
    if len(annotations) > n_frames:
        for i in range(n_frames, len(annotations)):
            del annotations[i]
        assert len(annotations) == n_frames
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
        n_frames = len(annotations)
    else:
        print("Loaded labeled dictionary successfully.")
else:
    print("Creating new dictionary.")
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
    global drawing, start_point, end_point, image, frame_no, annotations

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

        # Draw the last box on the original image
        image[:] = original_image[:]  # Clear the image to show only the last box
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

        # Save the box coordinates to annotations list
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
    cap.set(1, frame_no)  # Set the frame number to read
    ret, original_image = cap.read()
    if not ret:
        break  # Break if no more frames

    image = original_image.copy()  # Reset the image to original

    # Draw existing annotations for the current frame
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

    # Display instructions
    print(f"Frame {frame_no}/{n_frames - 1}: Draw a box with the mouse.")
    print(
        "Press\n's' to save the current frame, \n'n' for the next frame,\n'p' for the previous frame,\n'q' to quit."
    )

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):  # Save the current frame annotation
            # Save annotations to CSV
            save_anno_box(annotations, save_anno_name)
            break  # Break to process next frame

        elif key == ord("n"):  # Next frame
            frame_no += 1
            if frame_no >= n_frames:
                print("Reached the end of the video, back to 1st frame.")
                frame_no = 0
                # break
            break  # Break to load next frame
        elif key == ord("p"):
            frame_no -= 1
            if frame_no < 0:
                print("Reached the first frame, back to last frame.")
                frame_no = n_frames - 1
            break
        elif key == ord("f"):
            frame_no = 0
            break
        elif key == ord("l"):
            frame_no = n_frames - 1
            break
        elif key == ord(">"):
            frame_no = min(n_frames - 1, frame_no + 30)
            break
        elif key == ord("<"):
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

    if key == ord("n"):  # If moving to next frame, reset drawing
        drawing = False
        start_point = (-1, -1)
        end_point = (-1, -1)

cap.release()
cv2.destroyAllWindows()
