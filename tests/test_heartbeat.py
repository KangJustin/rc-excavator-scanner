import sys
import threading

import pytest

from reconstruct import config
from reconstruct.colmap import format_elapsed, run_command
from reconstruct.errors import ColmapCommandError


def sleeper(seconds, code=0, text="hello"):
    return [sys.executable, "-c", f"import time,sys; print('{text}'); time.sleep({seconds}); sys.exit({code})"]


def test_elapsed_formatting():
    assert format_elapsed(272) == "04:32" and format_elapsed(1127) == "18:47"
    assert format_elapsed(3725) == "1:02:05" and format_elapsed(0.4) == "00:00"


def test_heartbeat_lines_then_done(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 0.2)
    out = run_command(sleeper(0.75), "dense_reconstruction", tmp_path, tool="OpenMVS")
    lines = capsys.readouterr().out.splitlines()
    beats = [ln for ln in lines if "running..." in ln]
    assert len(beats) >= 2 and all(ln.startswith("[dense_reconstruction] running... 00:0") and ln.endswith("elapsed") for ln in beats)
    assert lines[-1] == "[dense_reconstruction] done in 00:00"
    assert out.strip() == "hello"  # tool output is untouched...
    assert (tmp_path / "dense_reconstruction.log").read_text().strip() == "hello"  # ...and the log has no heartbeat lines


def test_failure_keeps_exit_code_and_output_and_says_failed(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 0.2)
    with pytest.raises(ColmapCommandError) as e:
        run_command(sleeper(0.5, code=3, text="boom"), "mapper", tmp_path)
    assert e.value.returncode == 3 and "boom" in e.value.output and e.value.stage == "mapper"
    out = capsys.readouterr().out
    assert "[mapper] failed after 00:00" in out and "done in" not in out


def test_heartbeat_can_be_disabled_and_fast_commands_stay_quiet(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 0)
    run_command(sleeper(0.4), "x", tmp_path)
    assert "running..." not in capsys.readouterr().out
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 10)
    run_command(sleeper(0.0), "quick", tmp_path)
    out = capsys.readouterr().out
    assert "running..." not in out and "[quick] done in 00:00" in out


def test_no_thread_left_behind_even_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 0.1)
    before = threading.active_count()
    run_command(sleeper(0.3), "a", tmp_path)
    with pytest.raises(ColmapCommandError):
        run_command(sleeper(0.3, code=1), "b", tmp_path)
    assert threading.active_count() == before


def test_launch_failure_does_not_start_a_heartbeat(monkeypatch, capsys):
    from reconstruct.errors import ColmapNotFoundError
    monkeypatch.setattr(config, "HEARTBEAT_SECONDS", 0.1)
    with pytest.raises(ColmapNotFoundError):
        run_command(["/no/such/binary"], "s", None)
    assert capsys.readouterr().out == ""
