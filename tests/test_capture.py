import csv
from datetime import datetime

import numpy as np
import pytest

from capture import CaptureSession, FakeCamera, sharpness
from capture.__main__ import draw_overlay, run_preview
from capture.session import timestamp_name, unique_session_name


def test_timestamped_folder_name_format():
    assert timestamp_name(datetime(2026, 9, 25, 21, 30)) == "2026-09-25_2130"


def test_images_are_numbered_from_0001_directly_in_session_folder(tmp_path):
    s = CaptureSession(tmp_path, "2026-09-25_2130")
    cam = FakeCamera()
    for _ in range(3):
        s.save_frame(cam.read())
    assert sorted(p.name for p in s.dir.glob("*.jpg")) == ["0001.jpg", "0002.jpg", "0003.jpg"]
    assert s.count == 3


def test_csv_log_has_timestamp_and_index(tmp_path):
    s = CaptureSession(tmp_path, "t")
    cam = FakeCamera()
    for i in range(2):
        s.save_frame(cam.read(), waypoint=i)
    rows = list(csv.DictReader(open(s.log_path)))
    assert [r["index"] for r in rows] == ["1", "2"]
    assert rows[0]["file"] == "0001.jpg" and rows[0]["timestamp"] and float(rows[0]["sharpness"]) > 0
    meta = s.read_metadata()
    assert meta[1]["waypoint"] == 1 and meta[0]["index"] == 1 and isinstance(meta[0]["unix_time"], float)


def test_folder_is_created_lazily(tmp_path):
    s = CaptureSession(tmp_path)
    assert not s.dir.exists() and s.count == 0
    s.save_frame(FakeCamera().read())
    assert s.dir.is_dir()


def test_deleting_a_shot_never_causes_overwrite(tmp_path):
    s = CaptureSession(tmp_path, "t")
    cam = FakeCamera()
    for _ in range(3):
        s.save_frame(cam.read())
    (s.dir / "0002.jpg").unlink()
    assert s.save_frame(cam.read()).name == "0004.jpg"


def test_explicit_name_resumes_but_default_is_always_fresh(tmp_path):
    cam = FakeCamera()
    CaptureSession(tmp_path, "x").save_frame(cam.read())
    assert CaptureSession(tmp_path, "x").save_frame(cam.read()).name == "0002.jpg"
    a = CaptureSession(tmp_path)
    a.save_frame(cam.read())
    b = CaptureSession(tmp_path)  # same minute: must not reuse a's folder
    assert b.dir != a.dir and b.name.startswith(a.name)
    assert unique_session_name(tmp_path) not in (a.name,)


def test_sharpness_ranks_blur_lower():
    import cv2
    img = FakeCamera().read()
    assert sharpness(img) > sharpness(cv2.GaussianBlur(img, (0, 0), 6))


def test_overlay_draws_something_and_does_not_mutate_input():
    frame = FakeCamera().read()
    before = frame.copy()
    out = draw_overlay(frame, "s", 7, 100.0, 60.0, toast="saved 0007.jpg")
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    assert np.array_equal(frame, before)


def test_preview_loop_hotkeys(tmp_path):
    keys = iter([ord(" "), ord(" "), ord("n"), ord("n"), ord(" "), ord("q")])
    shown, said = [], []
    sessions = run_preview(FakeCamera(), str(tmp_path), None, 60.0, show=shown.append,
                           wait_key=lambda: next(keys), say=said.append)
    # 2 photos, N -> new session, N again while empty is ignored, 1 photo, quit
    assert [s.count for s in sessions] == [2, 1]
    assert len(sessions) == 2 and sessions[0].dir != sessions[1].dir
    assert len(shown) == 6
    assert sorted(p.name for p in sessions[0].dir.glob("*.jpg")) == ["0001.jpg", "0002.jpg"]
    assert sessions[1].count == 1 and (sessions[1].dir / "0001.jpg").exists()


def test_quitting_immediately_leaves_no_empty_folder(tmp_path):
    run_preview(FakeCamera(), str(tmp_path), None, 60.0, show=lambda _: None,
                wait_key=lambda: ord("q"), say=lambda _: None)
    assert list(tmp_path.iterdir()) == []


class FakeVideoCapture:
    """Stands in for cv2.VideoCapture: indices 0 (built-in) and 2 (USB) exist; 1 doesn't."""
    present = {0: (1280, 720), 2: (1280, 720)}

    def __init__(self, index):
        self.index = index

    def isOpened(self):
        return self.index in self.present

    def read(self):
        w, h = self.present[self.index]
        img = np.full((h, w, 3), 40 + 60 * self.index, np.uint8)
        return True, img

    def release(self):
        pass


def test_list_cameras_reports_each_and_saves_distinguishing_snapshots(tmp_path, monkeypatch):
    import cv2
    from capture.camera import list_cameras
    monkeypatch.setattr("capture.camera.cv2.VideoCapture", FakeVideoCapture)
    cams = list_cameras(max_index=4, snapshot_dir=tmp_path / "snaps")
    assert [(i, w, h) for i, w, h, _ in cams] == [(0, 1280, 720), (2, 1280, 720)]  # gap at 1 is fine
    a, b = (cv2.imread(str(s)) for *_, s in cams)
    assert a.shape == (720, 1280, 3) and abs(int(a.mean()) - int(b.mean())) > 30  # the images differ


def test_camera_flag_aliases_and_env_default(monkeypatch):
    import importlib
    import capture.__main__ as m

    seen = []

    def fake_check(args):
        seen.append(args.camera)
        return 0

    monkeypatch.setattr(m, "cmd_check", fake_check)
    for flag in ("--camera", "--camera-index", "--device"):
        assert m.main([flag, "2", "--check"]) == 0
    assert seen == [2, 2, 2]

    monkeypatch.setenv("CAMERA_INDEX", "3")  # the default is read when the parser is built
    assert m.main(["--check"]) == 0 and seen[-1] == 3
    monkeypatch.delenv("CAMERA_INDEX")
    assert m.main(["--check"]) == 0 and seen[-1] == 0


def test_overlay_shows_which_camera():
    frame = FakeCamera().read()
    assert not np.array_equal(draw_overlay(frame, "s", 1, 100.0, 60.0, camera_label="camera 2  1280x720"),
                              draw_overlay(frame, "s", 1, 100.0, 60.0))


def test_light_status_flags_dim_frames_and_passes_bright_ones():
    from capture import light_status
    dim = np.full((120, 160, 3), 60, np.uint8)
    bright = np.full((120, 160, 3), 140, np.uint8)
    assert light_status(dim) == (60.0, False) and light_status(bright) == (140.0, True)


def test_overlay_differs_for_a_dark_scene_by_more_than_just_the_pixels():
    """The dark warning must actually be drawn: same frame content, different brightness verdict."""
    from capture.__main__ import draw_overlay
    base = np.full((360, 640, 3), 100, np.uint8)
    bright, dark = base + 60, base - 40
    hot = draw_overlay(bright, "s", 3, 100.0, 60.0)[10:44, 300:640]  # the top-right label region
    cold = draw_overlay(dark, "s", 3, 100.0, 60.0)[10:44, 300:640]
    assert cold.shape == hot.shape and cold.std() > 0 and hot.std() > 0
