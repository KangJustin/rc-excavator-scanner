"""Pick the sharpest, evenly spread frames from a video and save them as photos.

    python -m capture.best_frames orbit.mp4                        # 50 photos -> best_frames/orbit/
    python -m capture.best_frames orbit.mp4 --count 40 --out scan_input/orbit
    python -m capture.best_frames orbit.mp4 --start 5 --end 65     # ignore the first 5 s and everything after 65 s

Why not just keep the N sharpest frames? They would bunch up wherever the camera moved slowly, and a scan needs
views from all the way round. So the video is cut into `--count` equal windows of time, the sharpest frame in each
window is the candidate, and a window whose best frame is still blurry is dropped instead of filled with a bad shot.

Blur is scored with the variance of the Laplacian (capture.camera.sharpness), the measure the camera check and the
sharpness readout already use. It only means something within one video, so "blurry" is relative: a window is
dropped when its best frame is below `--min-ratio` times the median of the windows' best frames (or below
`--min-sharpness`, if you set one).

The folder it writes is what `python -m reconstruct <folder>` takes: 0001.jpg, 0002.jpg, ... plus selection.csv
(which source frame and time each photo came from, and its sharpness).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import sharpness

LOW_SHARPNESS_WARN = 40.0   # capture --check calls a mean below this "low sharpness"
MIN_PHOTOS_WARN = 15        # reconstruct's preflight recommends at least this many
FALLBACK_FPS = 30.0         # some files report no frame rate
PHOTO_GLOB = "[0-9][0-9][0-9][0-9].jpg"
LISTED_DROPS = 5


class VideoError(RuntimeError):
    pass


@dataclass
class Scored:
    scores: np.ndarray      # sharpness of every decoded frame, in order
    fps: float
    fps_assumed: bool
    size: tuple[int, int]   # (width, height)


@dataclass
class Pick:
    frame: int              # index into the video's frames
    score: float
    window: int             # which of the windows it came from, 0-based


@dataclass
class Selection:
    picks: list[Pick]
    dropped: list[tuple[int, int, int]]   # (window, first frame, last frame) whose best frame was still blurry
    windows: int
    median: float                         # median of the windows' best sharpness
    threshold: float                      # a window's best frame has to reach this


def score_video(path: Path) -> Scored:
    """Sharpness of every frame, read once and not kept in memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    scores, size = [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            scores.append(sharpness(frame))
    finally:
        cap.release()
    if not scores:
        raise VideoError(f"No frames could be read from {path}. Is it a video file OpenCV can decode?")
    assumed = not fps > 0
    return Scored(np.array(scores), FALLBACK_FPS if assumed else fps, assumed, size)


def select_frames(scores, count: int, *, start: int = 0, end: int | None = None,
                  min_ratio: float = 0.5, min_sharpness: float = 0.0) -> Selection:
    """Cut frames [start, end) into `count` equal windows and keep the sharpest frame of each, unless even
    that frame is blurry (below max(min_sharpness, min_ratio * median of the windows' best))."""
    scores = np.asarray(scores, float)
    end = len(scores) if end is None else min(end, len(scores))
    if count < 1:
        raise ValueError("count must be at least 1")
    if not 0 <= start < end:
        raise ValueError("nothing to choose from: the start/end range holds no frames")
    windows = np.array_split(np.arange(start, end), min(count, end - start))
    best = [int(w[np.argmax(scores[w])]) for w in windows]
    median = float(np.median(scores[best]))
    threshold = max(min_sharpness, min_ratio * median)
    picks, dropped = [], []
    for n, (w, b) in enumerate(zip(windows, best)):
        if scores[b] < threshold:
            dropped.append((n, int(w[0]), int(w[-1])))
        else:
            picks.append(Pick(b, float(scores[b]), n))
    return Selection(picks, dropped, len(windows), median, threshold)


def extract_frames(path: Path, frames: list[int], out_dir: Path, quality: int = 95) -> list[Path]:
    """Write the given frame indices as 0001.jpg, 0002.jpg, ... (in frame order)."""
    photo_number = {f: n for n, f in enumerate(sorted(frames), 1)}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {path}")
    saved, index = [], 0
    try:
        while len(saved) < len(photo_number):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index in photo_number:
                dest = out_dir / f"{photo_number[index]:04d}.jpg"
                if not cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise VideoError(f"Failed to write {dest}")
                saved.append(dest)
            index += 1
    finally:
        cap.release()
    if len(saved) != len(photo_number):
        raise VideoError(f"Only {len(saved)} of {len(photo_number)} chosen frames could be read back; "
                         "the video decoded differently the second time. Try again, or re-encode it.")
    return saved


def _write_csv(path: Path, picks: list[Pick], files: list[Path], fps: float) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "file", "source_frame", "time_s", "sharpness", "window"])
        for n, (pick, file) in enumerate(zip(sorted(picks, key=lambda p: p.frame), files), 1):
            w.writerow([n, file.name, pick.frame, round(pick.frame / fps, 3), round(pick.score, 1), pick.window + 1])


def _fmt_time(frame: int, fps: float) -> str:
    return f"{frame / fps:.1f}s"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture.best_frames", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None, help="output folder (default: best_frames/<video name>)")
    p.add_argument("--count", type=int, default=50,
                   help="how many photos you want, and how many equal windows of time to cut the video into (default 50)")
    p.add_argument("--min-ratio", type=float, default=0.5,
                   help="drop a window whose best frame is below this share of the median best (default 0.5)")
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="also drop windows whose best frame is below this absolute Laplacian variance")
    p.add_argument("--start", type=float, default=0.0, help="ignore the video before this many seconds")
    p.add_argument("--end", type=float, default=None, help="ignore the video after this many seconds")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality of the saved photos (default 95)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the photos and selection.csv of an earlier run in --out (nothing else is deleted)")
    args = p.parse_args(argv)

    out = args.out or Path("best_frames") / args.video.stem
    earlier = sorted(out.glob(PHOTO_GLOB)) + ([out / "selection.csv"] if (out / "selection.csv").exists() else [])
    if earlier and not args.overwrite:
        print(f"ERROR: {out} already holds photos from an earlier run. Pick another --out, or pass --overwrite "
              "to replace them.", file=sys.stderr)
        return 2
    try:
        print(f"Reading {args.video} ...")
        video = score_video(args.video)
        start = round(args.start * video.fps)
        end = None if args.end is None else round(args.end * video.fps)
        sel = select_frames(video.scores, args.count, start=start, end=end,
                            min_ratio=args.min_ratio, min_sharpness=args.min_sharpness)
        if not sel.picks:
            raise VideoError(f"Every window's best frame was blurrier than the cut-off ({sel.threshold:.0f}), so "
                             "nothing was saved. Lower --min-ratio or --min-sharpness, or reshoot with more light "
                             "and slower motion.")
        for f in earlier:
            f.unlink()
        out.mkdir(parents=True, exist_ok=True)
        files = extract_frames(args.video, [pk.frame for pk in sel.picks], out, args.quality)
    except (VideoError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    _write_csv(out / "selection.csv", sel.picks, files, video.fps)

    n_frames = len(video.scores)
    lo = start
    hi = n_frames if end is None else min(end, n_frames)
    print(f"Video      : {args.video.name}  {video.size[0]}x{video.size[1]}, {video.fps:g} fps"
          f"{' (assumed, the file reports none)' if video.fps_assumed else ''}, {n_frames} frames "
          f"({n_frames / video.fps:.1f} s)")
    print(f"Considered : frames {lo} to {hi - 1} ({_fmt_time(lo, video.fps)} to {_fmt_time(hi - 1, video.fps)})")
    print(f"Sharpness  : median best {sel.median:.0f}, blur cut-off {sel.threshold:.0f}")
    print(f"Windows    : {sel.windows}  ->  kept {len(sel.picks)}, dropped {len(sel.dropped)}")
    for w, first, last in sel.dropped[:LISTED_DROPS]:
        print(f"             window {w + 1} ({_fmt_time(first, video.fps)} to {_fmt_time(last, video.fps)}): "
              "best frame still blurry")
    if len(sel.dropped) > LISTED_DROPS:
        print(f"             ... and {len(sel.dropped) - LISTED_DROPS} more")
    print(f"Saved      : {len(files)} photos in {out}/ ({files[0].name} ... {files[-1].name}) and selection.csv")

    kept_median = float(np.median([pk.score for pk in sel.picks]))
    if len(files) < MIN_PHOTOS_WARN:
        print(f"WARNING: only {len(files)} photos; reconstruction usually needs at least {MIN_PHOTOS_WARN}. "
              "Use a longer video, or a higher --count.", file=sys.stderr)
    if kept_median < LOW_SHARPNESS_WARN:
        print(f"WARNING: the kept photos have a median sharpness of {kept_median:.0f}, under the {LOW_SHARPNESS_WARN:.0f} "
              "the camera check calls low. That can mean the whole video is soft (add light, slow down, back off), or "
              "just a plain scene, since this measure depends on how much texture there is. Open a few photos to check.",
              file=sys.stderr)
    print(f"\nNext: python -m reconstruct {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
