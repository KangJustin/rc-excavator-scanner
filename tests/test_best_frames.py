import csv

import cv2
import numpy as np
import pytest

from capture.best_frames import VideoError, main, score_video, select_frames
from capture.camera import sharpness

W, H, FPS = 160, 120, 10


def make_video(path, n=60, blurred=(), fps=FPS):
    """n distinct textured frames; the indices in `blurred` are heavily blurred (as if the camera moved)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    assert writer.isOpened()
    for i in range(n):
        frame = np.random.default_rng(i).integers(0, 255, (H, W, 3), dtype=np.uint8)
        if i in blurred:
            frame = cv2.GaussianBlur(frame, (0, 0), 4)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------- the selection rule (no video)
def test_sharpest_frame_of_each_window_and_windows_spread_over_the_whole_video():
    got = select_frames([1, 5, 2, 9, 3, 4, 2, 8, 1], 3)
    assert [(p.frame, p.window) for p in got.picks] == [(1, 0), (3, 1), (7, 2)]
    assert got.dropped == [] and got.windows == 3


def test_a_window_that_is_all_blurry_is_dropped_not_filled_with_a_bad_shot():
    got = select_frames([10, 10, 10, 1, 2, 1, 10, 10, 10], 3)
    assert [p.window for p in got.picks] == [0, 2]
    assert got.dropped == [(1, 3, 5)]
    assert got.median == 10 and got.threshold == 5  # half of the median best


def test_absolute_floor_and_ratio_are_both_honoured():
    assert select_frames([6] * 6, 3, min_sharpness=20).picks == []
    got = select_frames([10, 4, 10, 4], 2, min_ratio=0.5)
    assert len(got.picks) == 2  # 10 and 10 are the bests, nothing is below the cut-off
    assert len(select_frames([10, 10, 3, 3], 2, min_ratio=0.5).dropped) == 1


def test_start_end_range_and_too_many_windows():
    got = select_frames(range(20), 100, start=4, end=8)
    assert [p.frame for p in got.picks] == [4, 5, 6, 7] and got.windows == 4  # one window per frame
    got = select_frames([0, 0, 0, 9, 0, 0, 7, 0], 2, start=2)
    assert [p.frame for p in got.picks] == [3, 6]  # the sharp frame at 0/1 is outside the range


def test_bad_arguments():
    with pytest.raises(ValueError, match="count"):
        select_frames([1, 2], 0)
    with pytest.raises(ValueError, match="no frames"):
        select_frames([1, 2], 1, start=5)


# ---------------------------------------------------------------- reading a real video file
def test_score_video_reads_every_frame_and_blur_scores_lower(tmp_path):
    v = score_video(make_video(tmp_path / "v.avi", n=12, blurred={3, 4}))
    assert len(v.scores) == 12 and v.fps == FPS and v.size == (W, H) and not v.fps_assumed
    assert v.scores[3] < 0.2 * v.scores[2] and v.scores[4] < 0.2 * v.scores[5]


def test_unreadable_video_is_an_error(tmp_path):
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    with pytest.raises(VideoError):
        score_video(tmp_path / "junk.mp4")
    with pytest.raises(VideoError, match="Could not open"):
        score_video(tmp_path / "missing.mp4")


# ---------------------------------------------------------------- the whole workflow
def rows(out):
    with open(out / "selection.csv") as f:
        return list(csv.DictReader(f))


def test_workflow_keeps_the_sharp_frames_and_drops_the_blurry_window(tmp_path, capsys):
    # 6 windows of 10 frames. Window 1 is blurry except its last frame (9); window 4 (frames 30-39) is all blurry.
    blurred = set(range(0, 9)) | set(range(30, 40))
    video = make_video(tmp_path / "orbit.avi", n=60, blurred=blurred)
    out = tmp_path / "shots"
    assert main([str(video), "--count", "6", "--out", str(out)]) == 0
    r = rows(out)
    assert [row["window"] for row in r] == ["1", "2", "3", "5", "6"]  # window 4 dropped
    assert [row["file"] for row in r] == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg", "0005.jpg"]
    assert int(r[0]["source_frame"]) == 9  # the one sharp frame in window 1, not one of the blurred ones
    assert all(int(row["source_frame"]) not in blurred for row in r)
    assert float(r[0]["time_s"]) == pytest.approx(0.9)
    photos = sorted(out.glob("*.jpg"))
    assert len(photos) == 5 and all(cv2.imread(str(p)).shape == (H, W, 3) for p in photos)
    # what was saved really is sharp: every photo scores far above a blurred frame
    blurred_score = score_video(video).scores[35]
    assert all(sharpness(cv2.imread(str(p))) > 5 * blurred_score for p in photos)
    text = capsys.readouterr()
    assert "kept 5, dropped 1" in text.out and "window 4" in text.out and "python -m reconstruct" in text.out
    assert "only 5 photos" in text.err  # under the 15 the reconstruction likes


def test_start_and_end_limit_which_part_of_the_video_is_used(tmp_path):
    video = make_video(tmp_path / "v.avi", n=60)
    out = tmp_path / "shots"
    assert main([str(video), "--count", "5", "--start", "2", "--end", "4", "--out", str(out)]) == 0
    assert all(20 <= int(row["source_frame"]) <= 39 for row in rows(out))


def test_second_run_needs_overwrite_and_only_replaces_its_own_files(tmp_path, capsys):
    video = make_video(tmp_path / "v.avi", n=40)
    out = tmp_path / "shots"
    assert main([str(video), "--count", "8", "--out", str(out)]) == 0
    (out / "notes.txt").write_text("mine")
    before = sorted(p.name for p in out.glob("*.jpg"))
    capsys.readouterr()
    assert main([str(video), "--count", "4", "--out", str(out)]) == 2  # refuses, changes nothing
    assert "--overwrite" in capsys.readouterr().err and sorted(p.name for p in out.glob("*.jpg")) == before
    assert main([str(video), "--count", "4", "--out", str(out), "--overwrite"]) == 0
    assert [p.name for p in sorted(out.glob("*.jpg"))] == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]
    assert len(rows(out)) == 4 and (out / "notes.txt").read_text() == "mine"


def test_nothing_is_saved_when_every_window_is_too_blurry(tmp_path, capsys):
    video = make_video(tmp_path / "v.avi", n=20, blurred=set(range(20)))
    out = tmp_path / "shots"
    assert main([str(video), "--count", "4", "--out", str(out), "--min-sharpness", "1e9"]) == 2
    assert "nothing was saved" in capsys.readouterr().err and not out.exists()


def test_missing_video_and_bad_range_are_errors_not_tracebacks(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")]) == 2
    assert "Could not open" in capsys.readouterr().err
    video = make_video(tmp_path / "v.avi", n=20)
    assert main([str(video), "--start", "50", "--out", str(tmp_path / "o")]) == 2
    assert "no frames" in capsys.readouterr().err


def test_default_output_folder_is_named_after_the_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = make_video(tmp_path / "my orbit.avi", n=20)
    assert main([str(video), "--count", "4"]) == 0
    assert len(list((tmp_path / "best_frames" / "my orbit").glob("*.jpg"))) == 4
