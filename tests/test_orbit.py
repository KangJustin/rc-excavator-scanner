import pytest

from orbit_capture import main, plan_waypoints


def test_full_circle_spacing():
    wps = plan_waypoints(20, 360)
    assert wps[0].step_deg == 0 and wps[1].step_deg == 18
    assert wps[-1].angle_deg == pytest.approx(342)  # doesn't double up on the start


def test_partial_arc_hits_both_ends():
    wps = plan_waypoints(5, 120)
    assert wps[-1].angle_deg == pytest.approx(120)


def test_end_to_end_dry_run(tmp_path, capsys):
    rc = main(["--dry-run", "--fake-camera", "--no-reconstruct", "--waypoints", "6", "--radius", "0.6",
               "--session-root", str(tmp_path), "--session-name", "s"])
    assert rc == 0
    imgs = sorted((tmp_path / "s").glob("*.jpg"))
    assert len(imgs) == 6
    out = capsys.readouterr().out
    assert out.count("SET_STICK left_y") == 5  # one arc per step after the first waypoint
    assert (tmp_path / "s" / "captures.csv").read_text().count("\n") == 7  # header + 6 rows


def test_orbit_hands_folder_to_reconstruction_and_reports_missing_colmap(tmp_path, capsys):
    rc = main(["--dry-run", "--fake-camera", "--waypoints", "4", "--radius", "0.6",
               "--colmap-bin", "no-such-colmap", "--session-root", str(tmp_path), "--session-name", "s"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "not found on PATH" in err and "safe in" in err
