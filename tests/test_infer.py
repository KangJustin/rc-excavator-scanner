import os
import signal

import numpy as np
import pytest

from record.infer import Runner, _raise_keyboard_interrupt, run_fake, threshold

CH = ("W", "O", "L", "A")


def test_threshold_handles_unrefined_model_output():
    # exactly the kind of raw values a cheap, undertrained checkpoint returns: small,
    # sometimes negative, never a clean 0/1 like the training data was.
    got = threshold([0.066, -0.023, 0.00094, 0.5, 0.499999])
    assert got.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert got.dtype == np.float32


def test_runner_never_calls_send_fn_when_not_driving():
    """The core safety property: without --drive, nothing reaches the hardware, no
    matter what the model says."""
    sent = []
    calls = {"n": 0}

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        calls["n"] += 1
        return np.ones((3, 4))  # "drive everything, hard" -- must still not be sent

    runner = Runner(get_frame, infer_fn, lambda ch, lv: sent.append((ch, lv)), CH,
                    chunk_size=3, max_steps=6, drive=False, sleep=lambda s: None)
    steps = runner.run()
    assert steps == 6 and calls["n"] == 2  # 6 steps / chunk_size 3 = 2 model calls
    assert sent == []  # nothing was ever sent, even though release_fn always fires


def test_runner_sends_thresholded_actions_when_driving():
    sent = []

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        return np.array([[1.0, -0.1, 0.51, 0.0]])  # one step, mixed values

    runner = Runner(get_frame, infer_fn, lambda ch, lv: sent.append((ch, lv)), CH,
                    chunk_size=1, max_steps=1, drive=True, sleep=lambda s: None)
    runner.run()
    sent_actions = [lv for ch, lv in sent if ch != "__release__"]
    assert dict(sent) == {"W": True, "O": False, "L": True, "A": False} or \
        sent[:4] == [("W", True), ("O", False), ("L", True), ("A", False)]


def test_state_becomes_the_previous_thresholded_action_not_the_raw_model_output():
    seen_states = []

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        seen_states.append(state.tolist())
        return np.array([[0.9, 0.1, 0.9, 0.1]])  # raw values thresholded to [1,0,1,0]

    runner = Runner(get_frame, infer_fn, lambda ch, lv: None, CH,
                    chunk_size=1, max_steps=3, drive=False, sleep=lambda s: None)
    runner.run()
    assert seen_states[0] == [0.0, 0.0, 0.0, 0.0]  # starts at zero, not measured
    assert seen_states[1] == [1.0, 0.0, 1.0, 0.0]  # the *thresholded* previous action
    assert seen_states[2] == [1.0, 0.0, 1.0, 0.0]


def test_chunk_smaller_than_requested_is_handled_and_triggers_a_fresh_call():
    """The real server can return fewer rows than execute_chunk_size asked for (we saw
    exactly 1 row come back on a real call). The loop must not crash or skip steps."""
    calls = {"n": 0}

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        calls["n"] += 1
        return np.zeros((1, 4))  # server always returns only 1 row, chunk_size asked for 5

    runner = Runner(get_frame, infer_fn, lambda ch, lv: None, CH,
                    chunk_size=5, max_steps=4, drive=False, sleep=lambda s: None)
    steps = runner.run()
    assert steps == 4 and calls["n"] == 4  # one model call per step, since chunks came back size 1


def test_everything_releases_even_when_the_camera_raises():
    released = []

    def get_frame():
        raise RuntimeError("camera stalled")

    runner = Runner(get_frame, lambda i, s: np.zeros((1, 4)), lambda ch, lv: None, CH,
                    chunk_size=1, max_steps=10, drive=True,
                    release_fn=lambda: released.append(True), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="camera stalled"):
        runner.run()
    assert released == [True]


def test_everything_releases_even_when_the_model_raises():
    released = []

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        raise ConnectionError("model unreachable")

    runner = Runner(get_frame, infer_fn, lambda ch, lv: None, CH,
                    chunk_size=1, max_steps=10, drive=True,
                    release_fn=lambda: released.append(True), sleep=lambda s: None)
    with pytest.raises(ConnectionError):
        runner.run()
    assert released == [True]


def test_default_release_fn_sets_every_channel_low_when_no_release_fn_given():
    sent = []

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    runner = Runner(get_frame, lambda i, s: np.ones((1, 4)), lambda ch, lv: sent.append((ch, lv)),
                    CH, chunk_size=1, max_steps=1, drive=True, sleep=lambda s: None)
    runner.run()
    # last 4 sends (the default release_fn) must set every channel low
    assert sent[-4:] == [("W", False), ("O", False), ("L", False), ("A", False)]


def test_record_fn_is_called_once_per_step_not_once_per_model_call():
    """The whole point of hooking record_fn into the inner per-step loop instead of
    alongside get_frame(): a saved video needs one frame per step tick, not one per
    model call (which would be `chunk_size` times sparser)."""
    calls = {"model": 0, "record": 0}

    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def infer_fn(image, state):
        calls["model"] += 1
        return np.zeros((3, 4))  # a full chunk each time

    runner = Runner(get_frame, infer_fn, lambda ch, lv: None, CH,
                    chunk_size=3, max_steps=6, drive=False, sleep=lambda s: None,
                    record_fn=lambda: calls.__setitem__("record", calls["record"] + 1))
    steps = runner.run()
    assert steps == 6 and calls["model"] == 2  # 6 steps / chunk_size 3 = 2 model calls
    assert calls["record"] == 6  # but recorded once per step, not once per model call


def test_max_steps_is_a_hard_stop():
    def get_frame():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    runner = Runner(get_frame, lambda i, s: np.ones((100, 4)), lambda ch, lv: None, CH,
                    chunk_size=100, max_steps=7, drive=False, sleep=lambda s: None)
    assert runner.run() == 7


# ---------------------------------------------------------------- SIGTERM safety
# A real incident: `kill` (SIGTERM) was used to stop a stuck run instead of Ctrl+C, and
# the excavator kept driving -- SIGTERM's default action skips all Python cleanup,
# unlike Ctrl+C (SIGINT), which Python's own default handler already turns into a
# catchable KeyboardInterrupt. main() now installs _raise_keyboard_interrupt for SIGTERM
# so a plain `kill` releases every line the same safe way Ctrl+C already did.
def test_sigterm_handler_raises_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        _raise_keyboard_interrupt(signal.SIGTERM, None)


def test_a_real_sigterm_mid_run_still_releases_every_line():
    """Sends this test process an actual SIGTERM while Runner.run() is executing --
    proving the real registration+delivery path releases the lines, not just that the
    handler function raises the right exception in isolation."""
    released = []
    old_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        def get_frame():
            os.kill(os.getpid(), signal.SIGTERM)
            return np.zeros((2, 2, 3), dtype=np.uint8)  # only reached if not interrupted

        runner = Runner(get_frame, lambda i, s: np.zeros((1, 4)), lambda ch, lv: None, CH,
                        chunk_size=1, max_steps=10, drive=True,
                        release_fn=lambda: released.append(True), sleep=lambda s: None)
        with pytest.raises(KeyboardInterrupt):
            runner.run()
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    assert released == [True]


# ---------------------------------------------------------------- run_fake (the CLI's own self-test)
def test_run_fake_needs_no_hardware_and_alternates_high_low():
    result = run_fake(max_steps=8, chunk_size=4, channel_names=CH)
    assert result["steps"] == 8 and result["model_calls"] == 2
    # first chunk (all-high) sent 16 writes (4 channels x 4 steps), then the release
    on_off = [lv for _ch, lv in result["sent"]]
    assert on_off[:4] == [True, True, True, True]  # first fake step, all channels high
    assert on_off[-4:] == [False, False, False, False]  # released at the very end
