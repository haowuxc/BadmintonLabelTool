## How to run
For new video

``` 
# label badminton
python imgLabel.py --label_video_path VIDEO_PATH

# label badminton racket
python imgLabelBat.py --label_video_path VIDEO_PATH

```

Continue label with existed csv file

```
# label badminton
python imgLabel.py --label_video_path VIDEO_PATH  --csv_path LABELED_CSV_PATH

# label badminton racket
python imgLabelBat.py --label_video_path VIDEO_PATH  --csv_path LABELED_CSV_PATH
```

## How to label
Mouse Event
- left click: label the center of ball (you can click many times for single frame, only the last position is keeped)
- middle/right click: cancel label of current frame 

Keyboard Event
- x: cancel label of current frame
- e: exit program
- s: save current label to a csv file
- o: go to next point (useful in racket labeling)
- q: go to previous point (useful in racket labeling)
- n: go to next frame
- p: go to previous frame
- f: go to the first frame
- l: go to the lasr frame
- \>:fast forward 36 frames
- <:fast backward 36 frames