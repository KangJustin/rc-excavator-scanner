"""`python -m reconstruct --doctor`: check the install before a real run.

Checks COLMAP (found? runs? with/without CUDA?), OpenMVS's two tools (found where you
pointed at? actually launch?), Open3D/OpenCV imports, then works out which dense backend
`auto` would really use. Nothing here touches images or writes files.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .colmap import Colmap
from .dense import AUTO_ORDER, OPENMVS_TOOLS, find_openmvs_tool

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
_TAG = {OK: "[ OK ]", INFO: "[info]", WARN: "[WARN]", FAIL: "[FAIL]"}

# Places a from-source OpenMVS install might have put its binaries. Only ever used to make
# a suggestion when the tools weren't found; never used to run anything.
_COMMON_OPENMVS_DIRS = ["/usr/local/bin/OpenMVS", "/opt/homebrew/bin/OpenMVS", "/usr/local/bin",
                        "~/openMVS/make/bin", "~/openMVS/build/bin", "~/openMVS/make/bin/Release"]
_SOURCE_CHECKOUTS = ["~/openMVS", "~/src/openMVS", "~/Developer/openMVS", "~/code/openMVS"]


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


@dataclass
class DoctorReport:
    settings: list[str]
    checks: list[Check] = field(default_factory=list)
    dense_pick: str | None = None  # "colmap" | "openmvs" | None (nothing can run dense)
    dense_notes: list[str] = field(default_factory=list)
    ready: bool = False  # a full run (sparse + dense + mesh) should be possible
    colmap_runs: bool = False

    def add(self, name, status, detail, fix=None):
        self.checks.append(Check(name, status, detail, fix))


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "(no output)")


def _find_below(root: Path, name: str, max_depth: int = 3) -> Path | None:
    """Depth-limited search for an executable called `name` under root."""
    root = root.expanduser()
    if not root.is_dir():
        return None
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        if name in filenames and os.access(Path(dirpath) / name, os.X_OK):
            return Path(dirpath) / name
    return None


def _suggest_openmvs_dir(name: str, configured: str) -> str | None:
    hits: list[Path | None] = []
    if configured:
        hits.append(_find_below(Path(configured), name))
    hits += [(Path(d).expanduser() / name) if (Path(d).expanduser() / name).is_file() else None
             for d in _COMMON_OPENMVS_DIRS]
    hits += [_find_below(Path(d), name, max_depth=5) for d in _SOURCE_CHECKOUTS]
    found = next((h for h in hits if h), None)
    return str(found.parent) if found else None


def _launches(exe: str) -> tuple[bool, str]:
    """Does the OpenMVS tool actually start? (catches missing shared libraries.)"""
    try:
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    out = r.stdout + r.stderr
    if r.returncode == 0 or "--help" in out:
        return True, ""
    return False, _first_line(out)


def run_doctor(colmap_bin: str, openmvs_bin_dir: str, dense_backend: str,
               import_module: Callable[[str], object] = importlib.import_module,
               on_macos: bool | None = None) -> DoctorReport:
    on_macos = (sys.platform == "darwin") if on_macos is None else on_macos
    rep = DoctorReport(settings=[
        f"platform        : {platform.platform()} ({platform.machine()}), Python {platform.python_version()}",
        f"COLMAP_BIN      : {colmap_bin}",
        f"OPENMVS_BIN_DIR : {openmvs_bin_dir or '(unset: search PATH)'}",
        f"DENSE_BACKEND   : {dense_backend}",
    ])

    # --- COLMAP -----------------------------------------------------------------
    colmap_runs, cuda = False, None
    exe = shutil.which(colmap_bin)
    if not exe:
        rep.add("COLMAP", FAIL, f"'{colmap_bin}' not found on PATH.",
                "macOS: `brew install colmap`. Linux: `sudo apt install colmap`. Or point at it with "
                "COLMAP_BIN in reconstruct/config.py.")
    else:
        probe = Colmap(colmap_bin).probe()
        if not probe.ran:
            rep.add("COLMAP", FAIL, f"found at {exe} but `colmap help` failed: {_first_line(probe.text)}",
                    "Reinstall it (a broken install / missing shared library is typical).")
        else:
            colmap_runs, cuda = True, probe.cuda
            rep.add("COLMAP", OK, f"{exe}  (version {probe.version or 'unrecognised'})")
            if probe.version is None:
                rep.add("COLMAP header", WARN, f"couldn't parse a version from: {_first_line(probe.text)}")
            if cuda is True:
                rep.add("COLMAP CUDA", OK, "header says: with CUDA")
            elif cuda is False:
                rep.add("COLMAP CUDA", INFO, "header says: without CUDA. COLMAP's dense stage can't run "
                        "with this build; sparse reconstruction is unaffected.")
            else:
                rep.add("COLMAP CUDA", INFO,
                        "header doesn't say with/without CUDA" + (
                            ". macOS builds have no CUDA, so COLMAP dense will fail here." if on_macos
                            else ". Can't tell up front; the dense stage will find out."))

    # --- OpenMVS ----------------------------------------------------------------
    openmvs_ok = True
    for tool in OPENMVS_TOOLS:
        path = find_openmvs_tool(tool, openmvs_bin_dir)
        if not path:
            openmvs_ok = False
            where = f"in '{openmvs_bin_dir}' or on PATH" if openmvs_bin_dir else "on PATH (OPENMVS_BIN_DIR is unset)"
            hint = _suggest_openmvs_dir(tool, openmvs_bin_dir)
            placeholder = ("  NOTE: '" + openmvs_bin_dir + "' looks like an EXAMPLE path copied from the docs, not a real folder. "
                           "Run `unset OPENMVS_BIN_DIR` (or export the real folder).") if "/path/to/" in openmvs_bin_dir else ""
            fix = (f"Found one at {hint}: run `export OPENMVS_BIN_DIR={hint}` or set it in reconstruct/config.py.{placeholder}"
                   if hint else
                   "Finish building OpenMVS (README > OpenMVS), then locate it with "
                   f"`find / -name {tool} -type f 2>/dev/null` and set OPENMVS_BIN_DIR to that folder.")
            rep.add(tool, FAIL, f"not found {where}.", fix)
            continue
        launches, why = _launches(path)
        if not launches:
            openmvs_ok = False
            rep.add(tool, FAIL, f"found at {path} but it won't run: {why}",
                    "Usually a missing shared library from the build (dyld/ld.so error above). "
                    "Rebuild, or install the missing dependency.")
        else:
            rep.add(tool, OK, path)

    # --- Python libraries -------------------------------------------------------
    open3d_ok = True
    for mod, label, why_needed in (("open3d", "Open3D", "meshing"),
                                   ("cv2", "OpenCV", "capture, photo checks and preview.png")):
        try:
            m = import_module(mod)
            rep.add(label, OK, f"imports (version {getattr(m, '__version__', '?')})")
        except Exception as e:  # ImportError, or a broken native library
            if mod == "open3d":
                open3d_ok = False
            rep.add(label, FAIL, f"import failed: {type(e).__name__}: {e}",
                    f"Needed for {why_needed}. `pip install -r requirements.txt` in your virtualenv.")

    # --- which dense backend would `auto` pick? ---------------------------------
    order = AUTO_ORDER if dense_backend == "auto" else (dense_backend,)
    if dense_backend not in ("auto", "colmap", "openmvs"):
        rep.dense_notes.append(f"DENSE_BACKEND '{dense_backend}' is not one of auto/colmap/openmvs.")
        order = ()
    for name in order:
        if name == "colmap":
            if not colmap_runs:
                rep.dense_notes.append("colmap: skipped (COLMAP isn't usable).")
            elif cuda is True:
                rep.dense_pick = "colmap"
                rep.dense_notes.append("colmap: header says with CUDA, so it will be used.")
            elif cuda is False or (cuda is None and on_macos):
                rep.dense_notes.append("colmap: no CUDA" + (" (macOS never has it)" if cuda is None else "")
                                       + ", so it can't do dense" + (
                                           "; auto will try it first, see it fail within seconds, and "
                                           "fall back." if dense_backend == "auto" else "."))
            else:  # cuda unknown, not macOS: give it a go
                rep.dense_pick = "colmap"
                rep.dense_notes.append("colmap: header doesn't mention CUDA; it will be attempted and, "
                                       "if it fails, auto falls back to OpenMVS.")
        elif name == "openmvs":
            if openmvs_ok:
                if rep.dense_pick is None:
                    rep.dense_pick = "openmvs"
                    rep.dense_notes.append("openmvs: both tools found and launch, so it will be used.")
                else:
                    rep.dense_notes.append("openmvs: available as the fallback.")
            else:
                rep.dense_notes.append("openmvs: not usable yet (see the FAIL lines above).")

    rep.colmap_runs = colmap_runs
    rep.ready = colmap_runs and open3d_ok and rep.dense_pick is not None
    return rep


def format_report(rep: DoctorReport) -> str:
    out = ["reconstruct --doctor", "", *rep.settings, ""]
    for c in rep.checks:
        out.append(f"{_TAG[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status in (FAIL, WARN):
            out.append(f"         fix: {c.fix}")
    out += ["", "Dense backend:"]
    out += [f"  - {n}" for n in rep.dense_notes]
    if rep.dense_pick:
        out.append(f"  => `auto` would use: {rep.dense_pick}")
    else:
        out.append("  => no dense backend can run right now."
                   + (" Sparse reconstruction would still work, but a full run stops at the dense stage."
                      if rep.colmap_runs else ""))
    out.append("")
    if rep.ready:
        out.append("READY: a full run (sparse -> dense -> mesh -> preview) should work.")
    else:
        out.append("NOT READY: fix the [FAIL] lines above, then run --doctor again.")
    return "\n".join(out)
