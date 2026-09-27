"""Tool locations and defaults. Edit these constants, or override per run with the
environment variables / command-line flags shown.

Once you know where COLMAP lives (and whether that machine has an NVIDIA GPU),
set COLMAP_BIN here and DENSE_BACKEND to "colmap" or "openmvs".
"""
import os

# Path to the COLMAP executable ("colmap" = look it up on PATH).      env: COLMAP_BIN   flag: --colmap-bin
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")

# Folder containing OpenMVS's InterfaceCOLMAP and DensifyPointCloud ("" = look them up
# on PATH). A from-source build leaves them in <openMVS build dir>/bin/<platform>/.
#                                                                     env: OPENMVS_BIN_DIR   flag: --openmvs-bin-dir
OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "/usr/local/bin/OpenMVS")  # where this machine's OpenMVS build installed

# Which engine makes the dense point cloud:
#   "colmap"  COLMAP patch_match_stereo   (needs an NVIDIA GPU + CUDA-enabled COLMAP)
#   "openmvs" OpenMVS DensifyPointCloud   (runs on CPU; slower)
#   "auto"    COLMAP if it can, else OpenMVS
#                                                                     env: DENSE_BACKEND   flag: --dense-backend
DENSE_BACKEND = os.environ.get("DENSE_BACKEND", "auto")

# Where reconstruction workspaces go (one sub-folder per session, named after it).
# Deliberately NOT inside the image folder: COLMAP scans image folders recursively and
# would pick up its own undistorted output on a re-run.
OUTPUT_ROOT = "reconstructions"

# Long external commands (dense reconstruction can take 30+ minutes) print a liveness line
# "[stage] running... MM:SS elapsed" this often, in seconds. They report no real progress,
# so it's only elapsed time. 0 turns the heartbeat off.
HEARTBEAT_SECONDS = 10.0
