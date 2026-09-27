import stat
import sys
from pathlib import Path

import pytest

from reconstruct.__main__ import main
from reconstruct.doctor import FAIL, INFO, OK, format_report, run_doctor

HERE = Path(__file__).parent


def wrapper(path: Path, script: str, *lead):
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / script}" {" ".join(lead)} "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def colmap_bin(tmp_path):
    return wrapper(tmp_path / "colmap", "fake_colmap.py")


@pytest.fixture
def openmvs_dir(tmp_path):
    d = tmp_path / "mvs"
    d.mkdir()
    wrapper(d / "InterfaceCOLMAP", "fake_openmvs.py", "InterfaceCOLMAP")
    wrapper(d / "DensifyPointCloud", "fake_openmvs.py", "DensifyPointCloud")
    return str(d)


@pytest.fixture(autouse=True)
def clean_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # so suggestion search can't see the real machine


def by_name(rep):
    return {c.name: c for c in rep.checks}


def test_mac_like_setup_without_cuda_picks_openmvs(colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    rep = run_doctor(colmap_bin, openmvs_dir, "auto", on_macos=True)
    c = by_name(rep)
    assert c["COLMAP"].status == OK and c["COLMAP CUDA"].status == INFO and "without CUDA" in c["COLMAP CUDA"].detail
    assert c["InterfaceCOLMAP"].status == OK and c["DensifyPointCloud"].status == OK
    assert c["Open3D"].status == OK
    assert rep.dense_pick == "openmvs" and rep.ready
    text = format_report(rep)
    assert "`auto` would use: openmvs" in text and "READY" in text and "OPENMVS_BIN_DIR" in text


def test_cuda_box_picks_colmap(colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "1")
    rep = run_doctor(colmap_bin, openmvs_dir, "auto", on_macos=False)
    assert "with CUDA" in by_name(rep)["COLMAP CUDA"].detail and rep.dense_pick == "colmap"
    assert any("available as the fallback" in n for n in rep.dense_notes)


def test_silent_header_on_macos_is_treated_as_no_cuda(colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "unknown")
    rep = run_doctor(colmap_bin, openmvs_dir, "auto", on_macos=True)
    assert "doesn't say" in by_name(rep)["COLMAP CUDA"].detail and "macOS" in by_name(rep)["COLMAP CUDA"].detail
    assert rep.dense_pick == "openmvs"


def test_colmap_missing(openmvs_dir):
    rep = run_doctor("no-such-colmap", openmvs_dir, "auto", on_macos=True)
    c = by_name(rep)["COLMAP"]
    assert c.status == FAIL and "brew install colmap" in c.fix
    assert not rep.ready  # sparse stage needs COLMAP
    assert "NOT READY" in format_report(rep)


def test_openmvs_missing_and_nothing_to_fall_back_on(colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    rep = run_doctor(colmap_bin, "", "auto", on_macos=True)
    c = by_name(rep)
    assert c["InterfaceCOLMAP"].status == FAIL and "OPENMVS_BIN_DIR is unset" in c["InterfaceCOLMAP"].detail
    assert rep.dense_pick is None and not rep.ready
    text = format_report(rep)
    assert "no dense backend can run" in text and "Sparse reconstruction would still work" in text
    # ...but with COLMAP missing too, don't claim sparse works
    assert "Sparse reconstruction would still work" not in format_report(run_doctor("nope", "", "auto", on_macos=True))


def test_wrong_openmvs_dir_suggests_the_right_one(colmap_bin, openmvs_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    parent = str(Path(openmvs_dir).parent)  # user pointed one level too high
    rep = run_doctor(colmap_bin, parent, "auto", on_macos=True)
    c = by_name(rep)["DensifyPointCloud"]
    assert c.status == FAIL and openmvs_dir in c.fix


def test_openmvs_that_wont_launch_is_reported(colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    monkeypatch.setenv("FAKE_OPENMVS", "broken")
    rep = run_doctor(colmap_bin, openmvs_dir, "auto", on_macos=True)
    c = by_name(rep)["DensifyPointCloud"]
    assert c.status == FAIL and "won't run" in c.detail and "Library not loaded" in c.detail
    assert not rep.ready


def test_open3d_import_failure(colmap_bin, openmvs_dir, monkeypatch):
    def imp(name):
        if name == "open3d":
            raise ImportError("no open3d")
        return __import__(name)
    monkeypatch.setenv("FAKE_CUDA", "0")
    rep = run_doctor(colmap_bin, openmvs_dir, "auto", import_module=imp, on_macos=True)
    assert by_name(rep)["Open3D"].status == FAIL and not rep.ready


def test_forced_backend_is_respected(colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    assert run_doctor(colmap_bin, openmvs_dir, "colmap", on_macos=True).dense_pick is None
    assert run_doctor(colmap_bin, openmvs_dir, "openmvs", on_macos=True).dense_pick == "openmvs"
    assert run_doctor(colmap_bin, openmvs_dir, "bogus", on_macos=True).dense_pick is None


def test_cli_doctor_needs_no_images_and_sets_exit_code(colmap_bin, openmvs_dir, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CUDA", "0")
    assert main(["--doctor", "--colmap-bin", colmap_bin, "--openmvs-bin-dir", openmvs_dir]) == 0
    assert "READY" in capsys.readouterr().out
    assert main(["--doctor", "--colmap-bin", "nope", "--openmvs-bin-dir", openmvs_dir]) == 1
    with pytest.raises(SystemExit):  # no --doctor and no folder
        main([])


def test_a_copied_placeholder_path_is_called_out_with_the_real_folder(colmap_bin, openmvs_dir, monkeypatch):
    """Regression: the README's example path '/path/to/openMVS/bin' was exported literally by a user."""
    monkeypatch.setenv("HOME", str(Path(openmvs_dir).parent))  # so the search below finds the fixture folder
    rep = run_doctor(colmap_bin, "/path/to/openMVS/bin", "auto", on_macos=True)
    c = by_name(rep)["DensifyPointCloud"]
    assert c.status == FAIL and "EXAMPLE path" in c.fix and "unset OPENMVS_BIN_DIR" in c.fix
