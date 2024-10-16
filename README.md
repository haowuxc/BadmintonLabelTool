## How to run
For new video

``` 
# label samll ball like tennis, badminton, etc.
python pointlabel.py --video_path VIDEO_PATH 

# label large ball like basketball, volleyball, etc.
python boxlabel.py --video_path VIDEO_PATH 

# label badminton racket
python imgLabelBat.py --label_video_path VIDEO_PATH



```

Continue label with existed csv file

```
# label samll ball like tennis, badminton, etc.
python pointlabel.py --video_path VIDEO_PATH --annotation_path LABELED_CSV_PATH

# label large ball like basketball, volleyball, etc.
python boxlabel.py --video_path VIDEO_PATH --annotation_path LABELED_CSV_PATH

# label badminton racket
python imgLabelBat.py --label_video_path VIDEO_PATH  --csv_path LABELED_CSV_PATH
```

## How to label
Status
- The blue text at the top left displays the current frame number and indicates whether the frame is labeled ('Labeled') or needs labeling ('To Label').

Mouse Event
- left click: label the center of ball (you can click many times for single frame, only the last position is keeped)
- middle/right click: cancel label of current frame 
- leftclick and drag to draw a box to label the ball


Keyboard Event for ball labeling.
- x: cancel label of current frame
- s: save current label to a csv file
- q: exit program
- n: go to next frame
- p: go to previous frame
- f: go to the first frame
- l: go to the lasr frame
- \>:fast forward 30 frames
- <:fast backward 30 frames

Keyboard Event for racket labeling
- x: cancel label of current frame
- e: exit program
- s: save current label to a csv file
- o: go to next point (useful in racket labeling)
- q: go to previous point (used in racket labeling)
- n: go to next frame
- p: go to previous frame
- f: go to the first frame
- l: go to the lasr frame
- \>:fast forward 36 frames
- <:fast backward 36 frames