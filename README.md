## How to run
For new video

``` python imgLabel.py --label_video_path VIDEO_PATH ```

Continue label with existed csv file

``` python imgLabel.py --label_video_path VIDEO_PATH  --csv_path LABELED_CSV_PATH```

## How to label
Mouse Event
- left click: label the center of ball (you can click many times for single frame, only the last position is keeped)
- middle click: cancel label of current frame 

Keyboard Event
- e: exit program
- s: save current label to a csv file
- n: go to next frame
- p: go to previous frame
- f: go to the first frame
- l: go to the lasr frame
- \>:fast forward 36 frames
- <:fast backward 36 frames