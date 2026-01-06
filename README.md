# Video Jump Detector

Detect jump cuts in a video using frame-difference energy plus robust Z-score
outlier detection. The detector can also export keyframes at detected jumps.

## Files

- `video_jump_detector.py`: core detector (batch).
- `video_jump_detector_progress.py`: same detector with live progress UI.
- `jump_detection_results.json`: example output.
- `video_files.txt`: record of ignored video filenames.
- `keyframes/`: exported keyframes (images).

## Requirements

- Python 3
- `opencv-python`
- `numpy`
- `tqdm`

## Usage

Process the first ~60 seconds of a video (edit `video_path` inside the script):

```bash
python3 video_jump_detector.py
```

With progress display:

```bash
python3 video_jump_detector_progress.py
```

Outputs:

- `jump_detection_results.json`
- `keyframes/*.jpg`
