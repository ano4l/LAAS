# Automatic pitch calibration (estimates for the shape view)

Status: first version, unvalidated. It feeds only the estimated pitch map in the Shape tab. It does not produce distances, speeds or heatmaps, and the measured fixed-camera path in `FIXED_CAMERA_CALIBRATION.md` is unchanged.

## What it does

For each analysed frame (every frame on a GPU, about five per second on a CPU):

1. A pose model marks up to 32 known pitch landmarks: line junctions, penalty spots, centre-circle points.
2. Landmarks with confidence of at least 0.5 that lie inside the image are kept.
3. A homography from image to pitch is fitted with RANSAC (1 m threshold) and refitted on the agreeing landmarks.
4. The frame is rejected unless all of these hold:
   - at least 5 landmarks agree, and they are at least 70% of those seen (four points fit exactly, so they cannot be cross-checked);
   - the landmarks span at least 9 m along both pitch axes and at least 1.5% of the image area;
   - mean landmark error is at most 0.6 m and the worst is at most 1.5 m;
   - the mapping is not mirrored and does not fold across the horizon.
5. A rejected frame has no pitch positions. Nothing is smoothed, carried over from another frame or interpolated.

Results are written to `pitch-calibration.json` in the run directory, with the per-frame matrix or the rejection reason. The shape snapshot for each second prefers a calibrated frame; if none qualifies, that second has a camera view only.

Player foot points (bottom centre of the detection box) are projected through the frame's own mapping. A player who lands more than 3 m outside the pitch outline is counted as not placed and is not drawn.

## What is assumed

- **Pitch size is not measured.** Touchline and goal-line landmarks assume 105 x 68 m. Markings with fixed sizes (penalty and goal areas, centre circle, penalty spots) follow the Laws of the Game. On a pitch of another size, positions near the edges shift.
- **Landmarks 11, 12, 19 and 20** are taken to be where the penalty arc meets the penalty-area line. This was read off the model's output on test frames, not from documentation.
- **No lens distortion model.** A homography cannot correct a wide-angle lens.

## What has and has not been checked

Checked, on one 30 s Bundesliga broadcast clip (the model's own training domain):

- 11 frames spread through the clip all calibrated, using 6 to 15 landmarks, with mean landmark error of 0.17 to 0.40 m.
- Drawn back onto the frames, the fitted lines follow the real markings near the landmarks. Away from them (near touchline, far end of the halfway line) the fitted lines were off by roughly 1 to 2 m by eye.

Not checked:

- Accuracy of player positions against any measured reference. The fit error describes agreement among landmarks, not position accuracy.
- Amateur footage, low cameras, faded or missing lines, non-standard pitches, wide-angle lenses. The model was trained on Bundesliga main-camera frames and should be expected to do worse elsewhere.
- Stability from frame to frame. Each frame is fitted alone, so positions can jitter between snapshots.
- The rejection thresholds. They are starting points chosen on the test frames, not tuned on labelled data.

## Model and licence

`football-pitch-detection.pt` from the `roboflow/sports` example setup script (YOLOv8x-pose, 32 keypoints, about 140 MB). It is not stored in Git; place it at `models/pitch-keypoints/football-pitch-detection.pt` or set `YOLO_PITCH_MODEL_PATH`. Its SHA-256 is recorded in each run's detection summary.

The model is an Ultralytics YOLO model, so it carries the same AGPL-3.0 terms as the detector already used here. The `roboflow/sports` code is MIT; the licence of the dataset the model was trained on was not confirmed.

## Next steps that would make the metres trustworthy

1. A labelled evaluation set with known positions, to measure real error.
2. Temporal modelling: solve the camera position once, then pan, tilt and zoom per frame, and smooth those rather than fitting every frame alone.
3. A lens distortion term.
4. Measured pitch dimensions per venue.
