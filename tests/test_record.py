import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from capture import FakeCamera
from control import make_excavator
from record.controls import (
    CONTROL_NAMES, KEY_DISCARD, KEY_NEUTRAL, KEY_SAVE, ControlState, apply_key, describe,
)
from record.episode import load_episode, run_episode
from record.pi import recording_window
from record.rawlog import RawLog, derive_actions, load_edges

FIXTURES = Path(__file__).parent / "fixtures"


def test_control_vector_roundtrip():
    s = apply_key(apply_key(ControlState(), ord("w")), ord("a"))
    v = s.to_vector()
    assert CONTROL_NAMES == ("W", "O", "L", "A")  # drive, arm up, camera tilt, steer
    assert v.dtype == np.float32 and v.shape == (4,) and v.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert ControlState.from_vector(v) == s
    with pytest.raises(ValueError, match="needs 4"):
        ControlState.from_vector([0.0, 0.0])  # e.g. an old/other-rig 2-value recording


def test_keys_toggle_independently_and_neutral_drops_everything():
    s = apply_key(ControlState(), ord("w"))
    assert s.level("W") and not any(s.level(n) for n in ("O", "L", "A"))
    s = apply_key(s, ord("a"))                  # steer is independent of drive: both can be high
    assert s.level("W") and s.level("A")
    off_w = apply_key(s, ord("w"))
    assert not off_w.level("W") and off_w.level("A")  # toggling W off leaves A on
    assert apply_key(s, KEY_NEUTRAL) == ControlState()
    assert apply_key(s, ord("?")) == s
    assert describe(s) == "W HIGH  O low  L low  A HIGH"


class FakeClock:
    def __init__(self):
        self.ns = 1_800_000_000_000_000_000

    def now(self):
        return self.ns

    def sleep(self, s):
        self.ns += int(s * 1e9)


def record(tmp_path, keys, fps=10, max_steps=None):
    clock = FakeClock()
    it = iter(keys)
    exc = make_excavator(dry_run=True)
    out = tmp_path / "ep"
    summary = run_episode(exc, FakeCamera(64, 48), out, task="orbit", fps=fps,
                          get_key=lambda f, s, n: next(it, KEY_SAVE),
                          max_steps=max_steps, clock_ns=clock.now, sleep=clock.sleep)
    return summary, out


def test_records_state_lagging_action_and_frames(tmp_path):
    summary, out = record(tmp_path, [ord("w"), -1, ord("a"), -1])  # drive, then also steer
    assert summary.steps == 4 and summary.achieved_fps == pytest.approx(10, rel=0.01)
    meta, steps = load_episode(out)
    assert meta["complete"] and meta["format_version"] == 2 and meta["control_names"] == ["W", "O", "L", "A"]
    assert len(steps) == 4 and all(s.frame_path.exists() for s in steps)
    assert [s.action.tolist() for s in steps] == [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]]
    assert [s.state.tolist() for s in steps] == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]]
    assert all(s.action.dtype == np.float32 and s.action.shape == (4,) for s in steps)
    assert all(b - a == 100_000_000 for a, b in zip([s.t_ns for s in steps], [s.t_ns for s in steps][1:]))
    assert summary.max_gap_s == pytest.approx(0.1)


def test_commands_reach_the_robot_and_end_neutral(tmp_path, capsys):
    record(tmp_path, [ord("w"), ord("a"), -1])
    out = capsys.readouterr().out
    assert "SET_LINE W 1" in out and "SET_LINE A 1" in out
    assert out.count("SET_LINE") == 2  # only changes are sent
    assert out.strip().splitlines()[-1].endswith("NEUTRAL")


def test_raw_log_records_start_level_and_every_edge_in_order(tmp_path):
    _, out = record(tmp_path, [ord("w"), -1, ord("o"), -1, KEY_NEUTRAL, -1])
    edges = load_edges(out / "controls_raw.jsonl")
    _, steps = load_episode(out)
    assert [(e["seq"], e["ch"], e["level"], e.get("init", False)) for e in edges] == [
        (0, "W", 0, True), (1, "O", 0, True), (2, "L", 0, True), (3, "A", 0, True),  # starting levels
        (4, "W", 1, False),                             # step 0
        (5, "O", 1, False),                             # step 2
        (6, "W", 0, False), (7, "O", 0, False),         # step 4: SPACE drops both on the same tick
    ]
    assert edges[0]["t_ns"] == steps[0].t_ns and edges[4]["t_ns"] == steps[0].t_ns
    assert edges[5]["t_ns"] == steps[2].t_ns and edges[6]["t_ns"] == edges[7]["t_ns"] == steps[4].t_ns


def test_actions_rederive_exactly_from_the_raw_log(tmp_path):
    _, out = record(tmp_path, [ord("w"), -1, ord("o"), -1, KEY_NEUTRAL, -1, ord("l")])
    _, steps = load_episode(out)
    derived = derive_actions(load_edges(out / "controls_raw.jsonl"), [s.t_ns for s in steps], 100_000_000)
    assert np.array_equal(derived, np.stack([s.action for s in steps]))


def test_discard_removes_episode(tmp_path):
    summary, out = record(tmp_path, [ord("w"), KEY_DISCARD])
    assert summary is None and not out.exists()


def test_max_steps_saves(tmp_path):
    summary, out = record(tmp_path, [-1] * 50, max_steps=5)
    assert summary.steps == 5 and json.loads((out / "meta.json").read_text())["steps"] == 5


def test_incomplete_episode_is_refused(tmp_path):
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "meta.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="not a complete"):
        load_episode(tmp_path / "e")


def test_recording_window_is_exact_and_end_exclusive(tmp_path):
    _, out = record(tmp_path, [-1, -1])
    _, steps = load_episode(out)
    start, end = recording_window(steps, 10)
    assert start.microsecond == (steps[0].t_ns // 1000) % 1_000_000
    assert (end - start).total_seconds() == pytest.approx(0.2)  # 2 steps at 10 Hz, end exclusive
    assert start.tzinfo is not None


# ---------------------------------------------------------------- raw log rule
MS = 1_000_000


def ev(t_ms, ch, level):
    return {"t_ns": t_ms * MS, "ch": ch, "level": level}


def test_derive_actions_high_for_at_least_half_the_step():
    # only W (index 0) is ever touched; O, L, A have no edges and must stay 0 throughout
    edges = [ev(20, "W", 1), ev(60, "W", 0),      # step 0: a 40 ms tap, under half of 100 ms -> 0
             ev(110, "W", 1), ev(170, "W", 0),    # step 1: 60 ms high, over half -> 1
             ev(190, "W", 1),                     # step 2: high from before the window to its end -> 1
             ev(350, "W", 0)]                     # step 3: high for 50 of 100 ms, exactly half -> 1
    got = derive_actions(edges, [0, 100 * MS, 200 * MS, 300 * MS], 100 * MS)
    assert got[:, 0].tolist() == [0, 1, 1, 1]
    assert got.dtype == np.float32 and got.shape == (4, 4) and got[:, 1:].sum() == 0
    assert derive_actions(edges, [], 100 * MS).shape == (0, 4)


def test_derive_actions_last_step_uses_one_period_and_same_time_edges_apply_in_order():
    edges = [ev(200, "W", 1), ev(200, "W", 0), ev(200, "O", 1)]  # W ends low, O ends high, at the same instant
    got = derive_actions(edges, [0, 100 * MS, 200 * MS], 100 * MS)
    assert got.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]


def test_raw_log_needs_start_first(tmp_path):
    with pytest.raises(RuntimeError):
        RawLog(tmp_path / "x.jsonl").update(0, ControlState())


# ---------------------------------------------------------------- stats / review
from record.review import (
    episode_stats, find_episodes, format_stats, is_rejected, legacy_episodes, review_episode, set_review,
)


def make_episode(root, name, keys):
    clock = FakeClock()
    it = iter(keys)
    run_episode(make_excavator(dry_run=True), FakeCamera(64, 48), root / name, task="orbit", fps=10,
                get_key=lambda f, s, n: next(it, KEY_SAVE), clock_ns=clock.now, sleep=clock.sleep)
    return root / name


def test_stats_idle_fraction_and_flags(tmp_path):
    ep = make_episode(tmp_path, "a", [ord("w"), -1, ord(" "), -1])  # steps: move, move, neutral, idle
    s = episode_stats(ep)
    assert s.steps == 4 and s.idle_fraction == 0.5 and s.longest_idle_s == pytest.approx(0.2)
    assert s.flags == ["SHORT"]  # under MIN_STEPS
    assert s.usable_steps == 2


def test_stats_report_counts_bad_and_incomplete(tmp_path):
    a = make_episode(tmp_path, "a", [ord("w")] + [-1] * 59)  # W goes high once and stays: no idle steps
    b = make_episode(tmp_path, "b", [ord("w")] + [-1] * 59)
    set_review(b, "bad")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "meta.json").write_text(json.dumps({"complete": False}))
    complete, incomplete = find_episodes(tmp_path)
    assert [p.name for p in complete] == ["a", "b"] and [p.name for p in incomplete] == ["c"]
    text = format_stats(tmp_path)
    assert "2 episodes, 120 steps" in text and "bad 1" in text and "incomplete" in text
    assert episode_stats(b).usable_steps == 0 and episode_stats(a).usable_steps == 60


def test_stats_empty_dir(tmp_path):
    assert "No complete episodes" in format_stats(tmp_path)


def test_review_marks_and_upload_filter_sees_it(tmp_path):
    ep = make_episode(tmp_path, "a", [ord("w")] * 10)
    shown = []

    def show(img):
        shown.append(img.shape)
        return ord("b") if len(shown) == 3 else -1

    assert review_episode(ep, show) == "bad" and len(shown) == 3
    assert is_rejected(ep)
    set_review(ep, "good")
    assert not is_rejected(ep)
    set_review(ep, None)
    assert "review" not in json.loads((ep / "meta.json").read_text())


def test_review_playback_pauses_at_end_and_supports_stepping(tmp_path):
    ep = make_episode(tmp_path, "a", [ord("w")] * 3)
    keys = iter([-1, -1, -1, -1, ord(","), ord("g")])  # plays 3 steps, holds last, steps back, then verdict
    calls = []

    def show(img):
        calls.append(1)
        return next(keys)

    assert review_episode(ep, show) == "good"
    assert json.loads((ep / "meta.json").read_text())["review"] == "good"
    assert review_episode(ep, lambda img: ord("n")) == "skip"
    assert review_episode(ep, lambda img: ord("q")) == "quit"


def test_frame_size_recorded_and_change_rejected(tmp_path):
    _, out = record(tmp_path, [-1, -1])
    assert json.loads((out / "meta.json").read_text())["frame_size"] == [64, 48]

    class Flaky(FakeCamera):
        def read_fresh(self):
            self.height = 48 if self._n == 0 else 40
            return super().read_fresh()

    clock, keys = FakeClock(), iter([-1] * 5)
    with pytest.raises(RuntimeError, match="size changed"):
        run_episode(make_excavator(dry_run=True), Flaky(64, 48), tmp_path / "x", task="t", fps=10,
                    get_key=lambda f, s, n: next(keys), clock_ns=clock.now, sleep=clock.sleep)


# ---------------------------------------------------------------- format guard, gaps
def test_old_or_mismatched_episodes_are_refused_and_reported(tmp_path):
    """Covers both real cases we've hit: no format_version at all (the pre-line stick/button
    format), and a format_version that matches but a different control_names (a 2-channel A/B
    recording from an earlier draft of this format -- exactly what a real teammate recording
    turned out to look like before the design settled on 4 lines)."""
    good = make_episode(tmp_path, "new", [ord("w"), -1])
    old = tmp_path / "old"
    old.mkdir()
    (old / "meta.json").write_text(json.dumps({"episode_id": "old", "complete": True, "steps": 1, "fps": 10}))
    with pytest.raises(ValueError, match="older format"):
        load_episode(old)

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    (mismatched / "meta.json").write_text(json.dumps({
        "episode_id": "mismatched", "complete": True, "steps": 0, "fps": 10,
        "format_version": 2, "control_names": ["A", "B"]}))
    with pytest.raises(ValueError, match="control_names"):
        load_episode(mismatched)

    assert find_episodes(tmp_path) == ([good], []) and set(legacy_episodes(tmp_path)) == {old, mismatched}
    text = format_stats(tmp_path)
    assert "1 episodes, 2 steps" in text and "mismatched control format, ignored" in text


def test_zero_step_episode_is_refused_with_a_clear_error(tmp_path):
    """A real field episode showed up complete:true with 0 steps and no frames (a crashed or
    interrupted recording that still got saved). Our own recorder never writes this shape (it
    deletes an episode with n==0), but a differently-behaved recorder can, so load_episode must
    reject it cleanly instead of the caller hitting an IndexError downstream in pi.py."""
    ep = _load_real_episode(tmp_path, "excavator-20260926-221851")
    with pytest.raises(ValueError, match="0 steps"):
        load_episode(ep)


def test_stats_flag_a_dropped_frame_but_keep_the_steps(tmp_path):
    ep = make_episode(tmp_path, "a", [ord("w")] + [-1] * 59)
    assert episode_stats(ep).flags == []
    shifted = []
    for line in (ep / "steps.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["i"] >= 30:
            d["t_ns"] += 300_000_000  # one 400 ms hole between steps 29 and 30
        shifted.append(json.dumps(d))
    (ep / "steps.jsonl").write_text("\n".join(shifted) + "\n")
    s = episode_stats(ep)
    assert s.max_gap_s == pytest.approx(0.4)
    assert s.flags == ["GAP"]  # the average rate is still inside the tolerance, so RATE is not raised
    assert s.usable_steps == 60  # flagged, not excluded


# ---------------------------------------------------------------- real recorded data (regression)
def _load_real_episode(tmp_path: Path, name: str) -> Path:
    """Copy a real episode's meta/steps/raw-log fixture into tmp_path and synthesize blank frames
    for every frame the steps reference, so tests exercise an actual field recording (real control
    traces, real timing) without storing real camera photos in the repo."""
    src, dest = FIXTURES / name, tmp_path / name
    dest.mkdir()
    for f in ("meta.json", "steps.jsonl", "controls_raw.jsonl"):
        (dest / f).write_bytes((src / f).read_bytes())
    meta = json.loads((dest / "meta.json").read_text())
    if meta.get("steps"):  # the empty fixture has no frame_size/frames worth creating
        w, h = meta["frame_size"]
        (dest / "frames").mkdir()
        for line in (dest / "steps.jsonl").read_text().splitlines():
            i = json.loads(line)["i"]
            cv2.imwrite(str(dest / "frames" / f"{i:06d}.jpg"), np.zeros((h, w, 3), dtype=np.uint8))
    return dest


def test_a_real_recorded_episode_loads_reviews_and_converts_to_shards(tmp_path):
    """excavator-20260926-222130: an actual episode recorded on the rig with control_names
    ["W","O","L","A"]. This is the file that caught the 2-channel/4-channel format mismatch."""
    ep = _load_real_episode(tmp_path, "excavator-20260926-222130")
    meta, steps = load_episode(ep)
    assert meta["control_names"] == ["W", "O", "L", "A"] and len(steps) == 101 == meta["steps"]
    assert all(s.action.shape == (4,) and s.action.dtype == np.float32 for s in steps)
    # ground truth from the real file: only W (drive) and A (steer) were ever used in this episode
    assert sorted(set(tuple(s.action.tolist()) for s in steps)) == [
        (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)]
    flags = episode_stats(ep).flags
    assert "GAP" in flags  # the real 211 ms gap this episode's meta reports
    assert "IDLE" in flags  # ~68% of it is all-zero, a real (if unideal) property of this episode

    n_shown = 0

    def show(img):
        nonlocal n_shown
        n_shown += 1
        return ord("g") if n_shown >= len(steps) else -1

    assert review_episode(ep, show) == "good"

    pytest.importorskip("pi_sdk.mcap")
    from record import pi
    paths = pi.write_shards(steps, pi.build_contract(meta["fps"]), tmp_path / "shards", meta["fps"])
    assert all(p.stat().st_size > 0 for p in paths)


# ---------------------------------------------------------------- PI side (SDK objects, no network)
def test_v2_contract_shapes_and_names():
    pytest.importorskip("pi_sdk")
    from record import pi
    c = {t.topic: t for t in pi.build_contract(10).topics}
    assert set(c) == {"system/step", pi.CAMERA_TOPIC, pi.STATE_TOPIC, pi.ACTION_TOPIC}
    assert c["system/step"].expected_frequency_hz == 10 and c["system/step"].frequency_tolerance == pi.STEP_TOLERANCE
    assert c[pi.STATE_TOPIC].ndarray.shape == (4,) and c[pi.ACTION_TOPIC].ndarray.shape == (4,)
    assert c[pi.ACTION_TOPIC].ndarray.dtype == "float32" and c[pi.ACTION_TOPIC].role == "action"
    assert pi.CONTRACT_NAME.endswith("_v2") and pi.IDS_FILE.name == "ids.v2.json"  # v1's .pi/ids.json is never touched


def test_setup_creates_profile_that_keeps_idle_steps_and_saves_to_the_v2_ids_file(tmp_path, monkeypatch):
    pytest.importorskip("pi_sdk")
    from types import SimpleNamespace
    from record import pi
    seen = {}

    class Data:
        def ensure_platform_contract(self, name, contract, description=""):
            seen["contract"] = name
            return SimpleNamespace(platform_contract=SimpleNamespace(platform_contract_id="c-1"))

    class Ft:
        def create_training_profile(self, name, contract_id, config):
            seen.update(profile=name, contract_id=contract_id, config=config)
            return SimpleNamespace(training_profile_id="p-1")

    monkeypatch.setattr(pi, "clients", lambda: (Data(), Ft()))
    monkeypatch.setattr(pi, "IDS_FILE", tmp_path / "ids.v2.json")
    ids = pi.setup(10)
    cfg = seen["config"]
    assert seen["contract"] == "rc_excavator_v2" and seen["contract_id"] == "c-1"
    assert cfg.keep_idle_actions is True and cfg.action_space == "absolute"
    assert cfg.base_camera_topics == (pi.CAMERA_TOPIC,) and cfg.action_topics == (pi.ACTION_TOPIC,)
    assert ids == {"platform_contract_id": "c-1", "training_profile_id": "p-1", "fps": 10}
    assert json.loads((tmp_path / "ids.v2.json").read_text()) == ids


def test_start_finetune_with_episode_ids_uses_a_dedicated_scoped_filter(tmp_path, monkeypatch):
    """Selecting specific episodes must not reuse FILTER_NAME (a dataset filter is
    immutable, and PI rejects reusing a name with a different query), and must filter on
    exactly those episode_ids, not the broad success/source_dataset_path query."""
    pytest.importorskip("pi_sdk")
    from types import SimpleNamespace
    from record import pi
    seen = {}

    class Ft:
        def create_dataset_filter(self, name, episode_filter):
            seen["filter_name"], seen["episode_filter"] = name, episode_filter
            return SimpleNamespace(dataset_filter_id="f-1")

        def create_finetune(self, name, filters, **kwargs):
            seen["run_name"], seen["filters"] = name, filters
            return SimpleNamespace(run_id="r-1")

    monkeypatch.setattr(pi, "clients", lambda: (None, Ft()))
    monkeypatch.setattr(pi, "load_ids", lambda: {"platform_contract_id": "c-1", "training_profile_id": "p-1"})
    monkeypatch.setattr(pi, "save_ids", lambda **kw: None)

    run_id = pi.start_finetune("my-run", num_steps=1000, episode_ids=("ep-a", "ep-b"))

    assert run_id == "r-1"
    assert seen["filter_name"] == f"{pi.FILTER_NAME}-my-run"  # never the broad, shared name
    flt = seen["episode_filter"]
    assert flt.platform_contract_id == "c-1"
    assert flt.episode_ids.include == ("ep-a", "ep-b")
    assert flt.success is None and flt.source_dataset_path is None  # scoped by id, not by these


def test_start_finetune_without_episode_ids_keeps_the_original_broad_filter(monkeypatch):
    """Backward compatible: no episode_ids means the same shared filter and query as before."""
    pytest.importorskip("pi_sdk")
    from types import SimpleNamespace
    from record import pi
    seen = {}

    class Ft:
        def create_dataset_filter(self, name, episode_filter):
            seen["filter_name"], seen["episode_filter"] = name, episode_filter
            return SimpleNamespace(dataset_filter_id="f-1")

        def create_finetune(self, name, filters, **kwargs):
            return SimpleNamespace(run_id="r-1")

    monkeypatch.setattr(pi, "clients", lambda: (None, Ft()))
    monkeypatch.setattr(pi, "load_ids", lambda: {"platform_contract_id": "c-1", "training_profile_id": "p-1"})
    monkeypatch.setattr(pi, "save_ids", lambda **kw: None)

    pi.start_finetune("my-run", num_steps=1000)
    assert seen["filter_name"] == pi.FILTER_NAME
    flt = seen["episode_filter"]
    assert flt.success is True and flt.source_dataset_path == pi.SOURCE_DATASET_PATH
    assert flt.episode_ids is None


def test_episode_converts_to_shards_that_satisfy_the_v2_contract(tmp_path):
    """v1.Writer checks every message against the contract as it writes, so this catches a shape
    or dtype mismatch locally, without touching PI."""
    pytest.importorskip("pi_sdk.mcap")
    from record import pi
    _, out = record(tmp_path, [ord("w"), -1, ord("o"), -1, KEY_NEUTRAL, -1])
    _, steps = load_episode(out)
    paths = pi.write_shards(steps, pi.build_contract(10), tmp_path / "shards", 10)
    assert [p.name for p in paths] == ["light.mcap", "camera.mcap"] and all(p.stat().st_size > 0 for p in paths)
