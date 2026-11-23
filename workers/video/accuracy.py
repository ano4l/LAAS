"""Small-object search geometry and explicit, reproducible analysis profiles."""
import math
import os


# profile: (analysis fps, detector image size). None analyses every source frame.
# balanced is the CPU setting between the two: its stride works out at 12.5, 10, 12.5 and 12 fps for 25, 30, 50 and 60 fps footage.
PROFILES = dict(accurate=(None, 1280), balanced=(12.5, 960), preview=(5, 640))


def analysis_settings(source_fps):
    profile = os.getenv("ANALYSIS_PROFILE", "accurate")
    if profile not in PROFILES:
        raise ValueError("ANALYSIS_PROFILE must be accurate, balanced or preview")
    default_fps, default_size = PROFILES[profile]
    fps = float(os.getenv("ANALYSIS_FPS", str(default_fps or source_fps)))
    size = int(os.getenv("YOLO_IMAGE_SIZE", str(default_size)))
    tile = int(os.getenv("BALL_TILE_SIZE", "0"))
    if not math.isfinite(fps) or fps <= 0 or size < 32 or size % 32 or tile < 0 or (tile and tile < 128):
        raise ValueError("FPS must be positive and finite; image size a multiple of 32; tile size 0 or at least 128")
    # Ceiling avoids silently exceeding a requested sampling rate.
    return dict(profile=profile, stride=max(1, math.ceil(source_fps / fps)), image_size=size, tile_size=tile)


def tile_windows(width, height, size, overlap=0.25):
    if size <= 0:
        return
    def starts(length):
        end = max(0, length - size)
        return sorted(set([*range(0, end + 1, max(1, int(size * (1 - overlap)))), end]))
    for y in starts(height):
        for x in starts(width):
            yield x, y, min(width, x + size), min(height, y + size)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def merge_ball_candidates(rows, iou_threshold=0.5):
    kept = []
    for row in sorted(rows, key=lambda r: r["confidence"], reverse=True):
        if not any(box_iou(row["bbox"], prior["bbox"]) >= iou_threshold for prior in kept):
            kept.append(row)
    return kept


def ball_search_window(width,height,last,velocity,frame_number,sample_fps):
    """A bounded inference crop; its predicted centre is never emitted as a detection."""
    if not last:return None
    elapsed=(frame_number-last[0])/sample_fps
    if not 0<elapsed<=.35:return None
    x=last[1]+velocity[0]*elapsed;y=last[2]+velocity[1]*elapsed
    if not 0<=x<width or not 0<=y<height:return None
    size=min(640,max(192,round(.25*min(width,height)+.4*math.hypot(width,height)*elapsed)))
    crop_w,crop_h=min(width,size),min(height,size)
    x1=max(0,min(width-crop_w,round(x-crop_w/2)))
    y1=max(0,min(height-crop_h,round(y-crop_h/2)))
    if crop_w==width and crop_h==height:return None
    return x1,y1,x1+crop_w,y1+crop_h
