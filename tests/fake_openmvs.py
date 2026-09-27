"""Stand-ins for OpenMVS's InterfaceCOLMAP and DensifyPointCloud (dispatch on argv[1]).
Mirror the real CLI as documented in the OpenMVS sources: InterfaceCOLMAP -i <colmap dir>
-o <scene.mvs> --image-folder <dir> -w <dir>;  DensifyPointCloud -i <scene.mvs> -o <out.mvs>
-w <dir>, which writes <out>.mvs AND <out>.ply.   FAKE_OPENMVS=fail makes densify exit non-zero."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fake_colmap import write_sphere_ply  # noqa: E402


def opt(args, *names):
    for n in names:
        if n in args:
            return args[args.index(n) + 1]
    raise SystemExit(f"missing option {names}")


tool, args = sys.argv[1], sys.argv[2:]
if os.environ.get("FAKE_OPENMVS") == "broken":  # e.g. a missing shared library
    print("dyld[1234]: Library not loaded: @rpath/libopencv_core.dylib"); sys.exit(1)
if "--help" in args:
    print("Generic options:\n  -h [ --help ]  produce this help message"); sys.exit(0)
if tool == "InterfaceCOLMAP":
    src = Path(opt(args, "-i", "--input-file"))
    assert (src / "sparse" / "images.bin").exists(), "input must contain sparse/"
    assert Path(opt(args, "--image-folder")).is_dir()
    Path(opt(args, "-o", "--output-file")).write_text("mvs")
elif tool == "DensifyPointCloud":
    if os.environ.get("FAKE_OPENMVS") == "fail":
        print("terminate called after throwing an instance of 'std::bad_alloc'"); sys.exit(134)
    assert Path(opt(args, "-i", "--input-file")).exists()
    out = Path(opt(args, "-o", "--output-file"))
    out.write_text("mvs")
    write_sphere_ply(out.with_suffix(".ply"))
else:
    sys.exit(f"unknown tool {tool}")
