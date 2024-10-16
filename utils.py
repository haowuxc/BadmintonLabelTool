import os

# import csv


def load_anno_box(csv_path):
    with open(csv_path, "r") as file:
        lines = file.readlines()
        n_frames = len(lines) - 1

        # Initialize info dictionary for each frame with default values for box coordinates
        info = {
            idx: {"Frame": idx, "Visibility": 0, "X1": -1, "Y1": -1, "X2": -1, "Y2": -1}
            for idx in range(n_frames)
        }

        # Loop through each line in the CSV (skipping the header)
        for line in lines[1:]:
            # Assuming the CSV format is: frame, Visibility, x1, y1, x2, y2
            frame, Visibility, x1, y1, x2, y2 = line.split(",")
            frame = int(frame)

            # Update info for the current frame
            info[frame]["Frame"] = frame
            info[frame]["Visibility"] = int(Visibility)
            info[frame]["X1"] = int(x1)
            info[frame]["Y1"] = int(y1)
            info[frame]["X2"] = int(x2)
            info[frame]["Y2"] = int(y2)

    return info


def load_anno_point(csv_path):
    with open(csv_path, "r") as file:
        lines = file.readlines()
        n_frames = len(lines) - 1

        # Initialize info dictionary for each frame with default values for box coordinates
        info = {
            idx: {"Frame": idx, "Visibility": 0, "X": -1, "Y": -1}
            for idx in range(n_frames)
        }

        # Loop through each line in the CSV (skipping the header)
        for line in lines[1:]:
            # Assuming the CSV format is: frame, Visibility, x1, y1, x2, y2
            frame, Visibility, x, y = line.split(",")
            frame = int(frame)

            # Update info for the current frame
            info[frame]["Frame"] = frame
            info[frame]["Visibility"] = int(Visibility)
            info[frame]["X"] = int(x)
            info[frame]["Y"] = int(y)

    return info


def save_anno_point(info, anno_path):
    success = False
    try:
        # Extract the video name (without the extension) to use as the CSV file name

        # Open the CSV file in write mode
        with open(anno_path, "w") as file:
            # Write the CSV header
            file.write("Frame,Visibility,x,y\n")

            # Loop through each frame in the info dictionary
            for frame in info:
                # Format the data with bounding box coordinates
                data = "{},{},{},{}".format(
                    info[frame]["Frame"],
                    info[frame]["Visibility"],
                    int(info[frame]["X"]),
                    int(info[frame]["Y"]),
                )
                # Write the formatted data to the CSV file
                file.write(data + "\n")

        success = True
        print("Save info successfully into", anno_path)

    except Exception as e:
        # Print the error message if something goes wrong
        print(f"Save info failure: {e}")

    return success


def save_anno_box(info, anno_path):
    success = False
    try:
        # Extract the video name (without the extension) to use as the CSV file name

        # Open the CSV file in write mode
        with open(anno_path, "w") as file:
            # Write the CSV header
            file.write("Frame,Visibility,x1,y1,x2,y2\n")

            # Loop through each frame in the info dictionary
            for frame in info:
                # Format the data with bounding box coordinates
                data = "{},{},{},{},{},{}".format(
                    info[frame]["Frame"],
                    info[frame]["Visibility"],
                    int(info[frame]["X1"]),
                    int(info[frame]["Y1"]),
                    int(info[frame]["X2"]),
                    int(info[frame]["Y2"]),
                )
                # Write the formatted data to the CSV file
                file.write(data + "\n")

        success = True
        print("Save info successfully into", anno_path)

    except Exception as e:
        # Print the error message if something goes wrong
        print(f"Save info failure: {e}")

    return success
