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

## Steadying: smoothing over time

A frame-by-frame homography follows the small errors of that frame's landmarks, so the whole map wobbles between frames, while the camera itself moves smoothly. `workers/video/pitch_camera.py`:

1. Follows where four fixed image points land on the pitch through each unbroken stretch of calibrated frames. A stretch ends at a gap longer than 0.5 s between calibrated frames, or where those points jump by more than 5 m between neighbouring samples and faster than 60 m/s (a cut).
2. Around each frame, fits a straight line through each point's path over about half a second (Gaussian weights, sigma 0.15 s, window 0.45 s either side, at least 5 calibrated frames) and reads it at that frame's time.
3. Rebuilds the frame's mapping from the four smoothed points.
4. Keeps the smoothed mapping only if it passes the same landmark checks a frame's own fit must pass (0.6 m mean, 1.5 m worst). Otherwise the frame keeps its own fit and is marked `stabilised: false`.

A steadied frame stores both mappings: `matrix` (smoothed, the one used) and `frame_matrix` (its own fit). Frames that were never calibrated are untouched: nothing is carried into them. The run summary records how many frames were steadied, their landmark agreement, and the frame-to-frame wobble of one fixed image point before and after. A calibration stored by an older version is steadied again from its stored frames the next time the run is opened; no model is needed.

### An approach that was tried and dropped

The first version fixed one camera position for the whole clip (the median of the positions implied by each frame's fit) and refitted rotation and zoom only. Fitted to each frame's own mapping it looked good. Checked against stored landmarks on three real clips it did not: the fixed-position mapping missed the landmarks by 0.7 to 1.3 m on average, against 0.2 to 0.3 m for a frame's own fit, so most frames failed the check and kept their own fit, and the mix made wobble worse on one clip. Refining the position by search, adding a shared lens-distortion term and freeing the principal point brought the error on the best clip from 17 px to about 5 px, still several times a free fit. The cause was not established; candidates are systematic landmark-model error, landmark definitions, and pitch size.

## Heat map, distance and speed (estimates)

`workers/video/pitch_motion.py` turns the per-frame positions into whole-clip figures for each kit group. They cover only the time a player was tracked in a calibrated frame, so they are not match totals.

- **Heat map.** The pitch is cut into 21 x 14 cells of about 5 x 4.9 m. Each calibrated sample adds its time to the cell each placed player stood in (player-seconds). A second grid records how long each cell was inside the camera's view, so a cell that was never seen is shown as unknown, not empty.
- **Distance and speed.** For each unbroken stretch of one track, positions are averaged over 0.5 s and distance is summed in 0.5 s hops. A hop faster than 11 m/s is treated as an error and left out (and counted). A missed sample, in tracking or in calibration, ends the stretch; nothing is bridged. The first and last quarter second of a stretch, where a full average does not fit, are not measured.
- **Per group:** summed distance and measured player-seconds over all its tracks, mean speed (distance divided by measured time), and the fastest half-second among tracks measured for at least 2 s.
- **Per tracked identity** (`players`): the same distance, speed and heat-map cells for each identity measured for at least 2 s, whether or not it was sorted into a kit group (the 60 longest-measured are kept). Group figures still count only observations sorted into that group.

The Players tab shows these as a player profile. A person can give a name and shirt number to one or more tracked identities (`workers/video/player_labels.py`, stored as `player-labels.json` beside the run). Nothing is recognised automatically: there is no face recognition and no shirt-number reading. Naming only groups identities and never changes an automatic figure. Saving is refused when two identities given to one player appear together in more than two analysed frames, because one person cannot be in two places.

The profile also shows:

- **Time at each speed.** Half-second hops are sorted into four bands (under 2, 2 to 4, 4 to 5.5, and 5.5 m/s and over). Half-second averages flatten short bursts, so the faster bands are undercounted.
- **Possibly the same player** (`workers/video/merge_suggestions.py`). Pairs of identities where one ends and another begins within 10 s and close enough for one person to have covered the ground (3 m plus 7 m per second of gap, at most 25 m), never two that share an analysed frame or sit in different kit groups. Without calibration at both ends the image is used instead, in player heights, over at most 3 s and without correcting for camera movement. Each suggestion counts its rivals. These are questions for a reviewer; nothing is joined automatically, and they have not been scored against labelled identities.
- **Names saved on other clips** are offered while typing (`GET /roster`). This only saves typing; nobody is matched across clips automatically.

Known weaknesses: a standing player still collects a little distance from noise; sharp turns inside a hop are cut short; a track is an unreviewed tracker identity, so one player can be several tracks and a track can swap players; the camera follows play, so the heat map favours wherever the camera pointed.

On the 30 s test clip the group figures were about 2.1 to 2.5 m/s average and 7.3 m/s for the fastest half-second, which is in the range expected of professional players, and they changed by about 2% between the per-frame and steadied mappings. That is a plausibility check, not a validation: there is no measured reference for that clip.

## What is assumed

- **Pitch size is not measured.** Touchline and goal-line landmarks assume 105 x 68 m. Markings with fixed sizes (penalty and goal areas, centre circle, penalty spots) follow the Laws of the Game. On a pitch of another size, positions near the edges shift.
- **Landmarks 11, 12, 19 and 20** are taken to be where the penalty arc meets the penalty-area line. This was read off the model's output on test frames, not from documentation.
- **No lens distortion model.** A homography cannot correct a wide-angle lens.
- **Steadying assumes the view changes smoothly** over about half a second. A whip pan or a hard zoom is left to each frame's own fit when the smoothed mapping fails the landmark checks.

## What has and has not been checked

Checked, on one 30 s Bundesliga broadcast clip (the model's own training domain):

- 11 frames spread through the clip all calibrated, using 6 to 15 landmarks, with mean landmark error of 0.17 to 0.40 m.
- Drawn back onto the frames, the fitted lines follow the real markings near the landmarks. Away from them (near touchline, far end of the halfway line) the fitted lines were off by roughly 1 to 2 m by eye.

- Steadying, against stored landmarks on three clips from the hosted library:

  | Clip | Calibrated frames | Steadied | Landmark error, own fit to steadied | Wobble per frame step, before to after |
  | --- | --- | --- | --- | --- |
  | 30 s, 1920 x 1080, 25 fps (Bundesliga) | 739 | 718 | 0.29 to 0.32 m | 0.13 to 0.03 m |
  | 1920 x 1080, 30 fps | 464 | 366 | 0.20 to 0.32 m | 0.66 to 0.47 m |
  | 848 x 480, 24 fps | 486 | 297 | 0.20 to 0.35 m | 0.96 to 0.80 m |

  The last two clips break into many short stretches (8 and 27), so many frames have too few calibrated neighbours and the wobble that remains is mostly at those breaks.
- On synthetic cameras with landmark noise, steadying more than halved the wobble and cut the position error near the touchline by over 30%. Synthetic data shares the method's assumptions, so this shows the code is right, not that real positions are.

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
