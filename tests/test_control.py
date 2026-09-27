import math

import pytest

from control import ButtonAxis, ControlError, ExcavatorConfig, StickAxis, make_excavator
from control.config import STICK_NEUTRAL


def dry(cfg=None):
    exc = make_excavator(dry_run=True, config=cfg)
    return exc, exc._t


def cmds(t):
    return [c for c in t.sent if c != "KEEPALIVE"]


def test_drive_arc_duration_and_sticks():
    cfg = ExcavatorConfig(drive_speed_mps=0.1, steer_yaw_rate_deg_s=30, stick_deadband=0.0,
                          drive_throttle=0.5)
    exc, t = dry(cfg)
    exc.drive_arc(0.5, 90)  # arc length = 0.5 * pi/2 = 0.785 m -> 7.85 s
    assert exc.dry_run_elapsed_s == pytest.approx(0.5 * math.pi / 2 / 0.1, abs=cfg.keepalive_s)
    # yaw needed = 0.1/0.5 = 0.2 rad/s vs 0.5236 max -> 38% steer; left turn, + = left
    steer = round(STICK_NEUTRAL + 0.2 / math.radians(30) * 127)
    assert cmds(t) == [f"SET_STICK left_y {round(128 + 0.5 * 127)}", f"SET_STICK left_x {steer}", "NEUTRAL"]
    assert "KEEPALIVE" in t.sent  # watchdog food while holding


def test_drive_arc_clockwise_inverts_steer():
    exc, t = dry(ExcavatorConfig(stick_deadband=0.0))
    exc.drive_arc(1.0, -45)
    steer_val = int(cmds(t)[1].split()[-1])
    assert steer_val < STICK_NEUTRAL


def test_drive_arc_too_tight_is_refused_before_moving():
    exc, t = dry()
    with pytest.raises(ControlError, match="tighter"):
        exc.drive_arc(0.01, 90)
    assert t.sent == []


def test_always_neutral_after_error():
    exc, t = dry()

    def boom(_):
        raise KeyboardInterrupt
    exc._sleep = boom
    with pytest.raises(KeyboardInterrupt):
        exc.drive_arc(1.0, 90)
    assert t.sent[-1] == "NEUTRAL"


def test_turret_uses_configured_stick_and_invert():
    exc, t = dry(ExcavatorConfig(turret_axis=StickAxis("right_x", invert=True), stick_deadband=0.0))
    exc.rotate_turret(90)  # 90 deg / 20 deg/s = 4.5 s
    assert exc.dry_run_elapsed_s == pytest.approx(4.5, abs=0.2)
    assert cmds(t) == ["SET_STICK right_x 1", "NEUTRAL"]


def test_button_press_is_chunked():
    exc, t = dry(ExcavatorConfig(max_press_ms=1000))
    exc.press_button("boom_up", 2500)
    presses = [c for c in cmds(t) if c.startswith("PRESS")]
    assert presses == ["PRESS_BUTTON boom_up 1000", "PRESS_BUTTON boom_up 1000", "PRESS_BUTTON boom_up 500"]


def test_arm_needs_homing_then_dead_reckons():
    exc, t = dry(ExcavatorConfig(arm_full_travel_s=2.0, arm_axis=ButtonAxis("boom_down", "boom_up")))
    with pytest.raises(ControlError, match="unknown"):
        exc.move_arm(0.5)
    exc.home_arm()
    assert exc.arm_position == 0.0
    t.sent.clear()
    exc.move_arm(0.5)  # + direction = plus button, 0.5 * 2 s = 1 s
    assert cmds(t)[0] == "PRESS_BUTTON boom_down 1000"
    assert exc.arm_position == 0.5


def test_config_roundtrip_and_validation(tmp_path):
    cfg = ExcavatorConfig(turret_axis=ButtonAxis("a", "b"), drive_speed_mps=0.3)
    assert ExcavatorConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ExcavatorConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        StickAxis("nope")


def test_real_transport_requires_port():
    with pytest.raises(ControlError, match="--dry-run"):
        make_excavator(port=None, dry_run=False)


def test_set_line_sends_the_wire_command_and_rejects_unknown_lines():
    exc, t = dry()
    exc.set_line("W", True)
    exc.set_line("O", False)
    exc.set_line("W", False)
    assert t.sent == ["SET_LINE W 1", "SET_LINE O 0", "SET_LINE W 0"]
    with pytest.raises(ValueError, match="unknown line"):
        exc.set_line("Z", True)
    assert len(t.sent) == 3  # nothing was sent for the bad name


def test_set_line_cli_holds_then_returns_to_neutral(capsys):
    from control.__main__ import main
    assert main(["--dry-run", "set-line", "L", "1", "--hold", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "t=   0.00s] SET_LINE L 1" in out
    assert "t=   0.40s] NEUTRAL" in out  # released only after the 0.4 s hold
    assert out.index("SET_LINE L 1") < out.index("NEUTRAL")
