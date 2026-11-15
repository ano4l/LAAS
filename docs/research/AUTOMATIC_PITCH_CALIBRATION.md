# Automatic pitch calibration (estimates for the shape view)

Status: first version, unvalidated. It feeds the estimated pitch map, heat map, distance and speed figures in the Shape tab, all labelled as estimates. The measured fixed-camera path in `FIXED_CAMERA_CALIBRATION.md` is unchanged.

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
5. A rejected frame has no pitch positions. Nothing is carried into it from another frame or interpolated.
6. The calibrated frames are then steadied with one camera for the clip, where they agree on one (next section).

Results are written to `pitch-calibration.json` in the run directory, with the per-frame matrix or the rejection reason, and the image positions of the landmarks used. The shape snapshot for each second prefers a calibrated frame; if none qualifies, that second has a camera view only.

Player foot points (bottom centre of the detection box) are projected through the frame's own mapping. A player who lands more than 3 m outside the pitch outline is counted as not placed and is not drawn.

## Steadying: one camera for the clip

A frame-by-frame homography has eight free numbers, so small landmark errors move the whole map between frames. A broadcast camera stands in one place and only turns and zooms. `workers/video/pitch_camera.py`:

1. Reads a camera position and focal length out of each calibrated frame's own fit (pinhole camera, square pixels, principal point at the image centre).
2. Takes the median position as the clip's camera. If fewer than 8 frames give a position, or the positions scatter by more than 20% of the camera's distance to the pitch centre (for example a clip that mixes cameras), nothing is steadied and every frame keeps its own fit.
3. Refits each frame with the position fixed: rotation and zoom only, four numbers instead of eight.
4. Smooths rotation and zoom with a local straight-line fit over about half a second (Gaussian weights, sigma 0.15 s, window 0.45 s either side). Smoothing never crosses a gap longer than 0.5 s between calibrated frames, a turn of more than 4 degrees or a zoom change of more than 15% between neighbouring samples.
5. Keeps the steadied mapping for a frame only if it still matches that frame's landmarks to 1.0 m on average and 2.5 m at worst. Otherwise the frame keeps its own fit and is marked `stabilised: false`.

A steadied frame stores both mappings: `matrix` (steadied, the one used) and `frame_matrix` (its own fit). Frames that were never calibrated are untouched. The run summary records the camera position, how many frames were steadied, and the frame-to-frame wobble of one fixed image point before and after.

## Heat map, distance and speed (estimates)

`workers/video/pitch_motion.py` turns the per-frame positions into whole-clip figures for each kit group. They cover only the time a player was tracked in a calibrated frame, so they are not match totals.

- **Heat map.** The pitch is cut into 21 x 14 cells of about 5 x 4.9 m. Each calibrated sample adds its time to the cell each placed player stood in (player-seconds). A second grid records how long each cell was inside the camera's view, so a cell that was never seen is shown as unknown, not empty.
- **Distance and speed.** For each unbroken stretch of one track, positions are averaged over 0.5 s and distance is summed in 0.5 s hops. A hop faster than 11 m/s is treated as an error and left out (and counted). A missed sample, in tracking or in calibration, ends the stretch; nothing is bridged. The first and last quarter second of a stretch, where a full average does not fit, are not measured.
- **Per group:** summed distance and measured player-seconds over all its tracks, mean speed (distance divided by measured time), and the fastest half-second among tracks measured for at least 2 s.

Known weaknesses: a standing player still collects a little distance from noise; sharp turns inside a hop are cut short; a track is an unreviewed tracker identity, so one player can be several tracks and a track can swap players; the camera follows play, so the heat map favours wherever the camera pointed.

On the 30 s test clip the group figures were about 2.1 to 2.5 m/s average and 7.3 m/s for the fastest half-second, which is in the range expected of professional players, and they changed by about 1% between the per-frame and steadied mappings. That is a plausibility check, not a validation: there is no measured reference for that clip.

## What is assumed

- **Pitch size is not measured.** Touchline and goal-line landmarks assume 105 x 68 m. Markings with fixed sizes (penalty and goal areas, centre circle, penalty spots) follow the Laws of the Game. On a pitch of another size, positions near the edges shift.
- **Landmarks 11, 12, 19 and 20** are taken to be where the penalty arc meets the penalty-area line. This was read off the model's output on test frames, not from documentation.
- **No lens distortion model.** A homography cannot correct a wide-angle lens.
- **Steadying assumes one fixed camera position per clip**, turning and zooming only, with no roll change being special-cased (roll is fitted freely and smoothed).

## What has and has not been checked

Checked, on one 30 s Bundesliga broadcast clip (the model's own training domain):

- 11 frames spread through the clip all calibrated, using 6 to 15 landmarks, with mean landmark error of 0.17 to 0.40 m.
- Drawn back onto the frames, the fitted lines follow the real markings near the landmarks. Away from them (near touchline, far end of the halfway line) the fitted lines were off by roughly 1 to 2 m by eye.

- Steadying, on the same clip's 188 stored calibrations (landmark positions were not stored for that run, so it was fitted to each frame's own mapping): one camera was found, on the halfway line about 38 m behind the near touchline and 17 m above the pitch, with a position scatter of 4%. All 188 frames were steadied. The wobble of a fixed image point fell from 0.22 m to 0.06 m per frame step. The steadied mapping moved positions by 0.64 m on average (1.2 m in the worst frame) from the frame's own fit. Drawn on three frames, the two sets of lines were equally close to the real markings by eye; neither was clearly better.
- On synthetic cameras with landmark noise, steadying halved both the wobble and the position error near the touchline. Synthetic data shares the method's assumptions, so this shows the code is right, not that real positions are.

Not checked:

- Distance, speed and heat map values against any tracking-system reference.
- Whether steadying makes real positions more accurate. Only that it makes them steadier.
- Steadying on a clip with replays, cuts or a moving camera.

- Accuracy of player positions against any measured reference. The fit error describes agreement among landmarks, not position accuracy.
- Amateur footage, low cameras, faded or missing lines, non-standard pitches, wide-angle lenses. The model was trained on Bundesliga main-camera frames and should be expected to do worse elsewhere.
- The rejection thresholds. They are starting points chosen on the test frames, not tuned on labelled data.

## Model and licence

`football-pitch-detection.pt` from the `roboflow/sports` example setup script (YOLOv8x-pose, 32 keypoints, about 140 MB). It is not stored in Git; place it at `models/pitch-keypoints/football-pitch-detection.pt` or set `YOLO_PITCH_MODEL_PATH`. Its SHA-256 is recorded in each run's detection summary.

The model is an Ultralytics YOLO model, so it carries the same AGPL-3.0 terms as the detector already used here. The `roboflow/sports` code is MIT; the licence of the dataset the model was trained on was not confirmed.

## Next steps that would make the metres trustworthy

1. A labelled evaluation set with known positions, to measure real error.
2. Use the clip camera to recover frames with too few landmarks for a fit of their own (two landmarks are enough once the position is known).
3. A lens distortion term.
4. Measured pitch dimensions per venue.
