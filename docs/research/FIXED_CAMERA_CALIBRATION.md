# Fixed-camera calibration foundation

This CLI produces optional observed player movement in metres after explicit geometry and camera checks. It does not yet have a dashboard editor. No measured calibration has been supplied for the test videos.

Create a JSON file for the exact source video:

```json
{
  "source_sha256": "REPLACE_WITH_THE_SOURCE_SHA256_FROM_MANIFEST",
  "camera_mode": "fixed",
  "pitch_length_m": 100,
  "pitch_width_m": 60,
  "points": [
    {"image_xy": [100, 100], "pitch_xy_m": [0, 0]},
    {"image_xy": [900, 100], "pitch_xy_m": [100, 0]},
    {"image_xy": [900, 500], "pitch_xy_m": [100, 60]},
    {"image_xy": [100, 500], "pitch_xy_m": [0, 60]},
    {"image_xy": [500, 100], "pitch_xy_m": [50, 0]},
    {"image_xy": [500, 500], "pitch_xy_m": [50, 60]}
  ]
}
```

**These numbers describe an artificial rectangular test image. Replace every coordinate and both dimensions with measured values for the real source. Do not apply the example to football footage.** Coordinates in `image_xy` refer to original-resolution pixels, not the resized annotated export. Pitch coordinates use a consistent origin and measured dimensions in metres. Spread fitting points across the visible ground plane; the last two points are held out from fitting. At least four fitting points and two held-out points are required; more points are preferable.

```powershell
node scripts/python.mjs scripts/calibrate_tracking.py path/to/source.mp4 path/to/run-directory path/to/calibration.json
```

The CLI verifies source hash, landmark shape/bounds/uniqueness, useful fitting area, held-out world error (at most 0.75m) and reprojection error (at most 3px). It checks background features throughout the decoded source. Camera movement beyond the small tolerance, insufficient correspondences, or incomplete decoding prevents physical output. Broadcast pans/zooms require a different per-view calibration path.

`calibrated-metrics.json` contains `available`, the camera check, calibration diagnostics and per-track accepted observed distance, transition seconds, mean observed speed and excluded-step counts. Only player foot points are projected. Aerial balls are not mapped onto the ground. Steps above 15m/s and gaps above 0.2s are excluded. Detection jitter, footpoint uncertainty and identity errors still affect results; validate against independent measurements before use. This is not complete workload or peak speed.

A homography does not correct wide-angle/fisheye lens distortion. Such cameras may require measured intrinsics and undistortion before landmark annotation and detection; both must use the same corrected image coordinates. This CLI does not perform lens correction.
