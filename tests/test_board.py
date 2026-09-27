import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from measure import board as B

SHIPPED = Path(__file__).parent.parent / "scale_board"


def test_board_fits_both_paper_sizes_with_printer_safe_margins():
    w, h = B.board_size_mm()
    assert (w, h) == (150.0, 210.0)
    for name, (pw, ph) in B.PAPERS_MM.items():
        assert (pw - w) / 2 >= 20 and B.TOP_MARGIN_MM + h + 45 <= ph, name  # side margins; room for ruler + text


def test_render_grid_puts_every_edge_on_a_pixel():
    assert (B.PX_PER_SQUARE * B.BOARD["marker_mm"] / B.BOARD["square_mm"]) % 6 == 0  # 6 cells per tag


def test_vector_rectangles_reproduce_the_bitmap_exactly():
    img = B.render_pixels()
    rects = B.black_rects(img)
    assert np.array_equal(B.rasterize_rects(rects, img.shape), img)
    assert len(rects) < 2000  # merged, not one rect per pixel


def test_pdf_is_structurally_valid_and_page_size_is_exact():
    for paper, (w, h) in B.PAPERS_MM.items():
        data = B.build_pdf(paper)
        assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
        xref = int(re.search(rb"startxref\n(\d+)", data).group(1))
        assert data[xref:xref + 4] == b"xref"
        offsets = re.findall(rb"(\d{10}) 00000 n", data)
        for i, off in enumerate(offsets, 1):  # every xref entry must point at its "N 0 obj"
            assert data[int(off):].startswith(b"%d 0 obj" % i)
        mb = [float(x) for x in re.search(rb"/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]", data).groups()]
        assert mb == pytest.approx([w * 72 / 25.4, h * 72 / 25.4], abs=0.001)


def test_shipped_files_match_the_single_source_of_truth():
    """If someone edits BOARD without regenerating scale_board/, the printed sheet and the
    software would disagree about the size. Fail loudly."""
    for paper in B.PAPERS_MM:
        assert (SHIPPED / f"charuco_{paper}.pdf").read_bytes() == B.build_pdf(paper), f"regenerate: python -m measure.board"
    spec = json.loads((SHIPPED / "charuco_board.json").read_text())
    assert all(spec[k] == v for k, v in B.BOARD.items())
    assert spec["board_width_mm"] == 150 and spec["inner_corners"] == 24


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm (poppler) not installed")
@pytest.mark.parametrize("paper", ["letter", "a4"])
def test_printed_pdf_detects_and_measures_30mm_squares(paper, tmp_path):
    """Render the real PDF at a known DPI, detect the board in it, and measure the squares."""
    dpi = 300
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(SHIPPED / f"charuco_{paper}.pdf"),
                    str(tmp_path / "p")], check=True)
    img = cv2.imread(str(tmp_path / "p.png"))
    cc, ci, mc, mi = cv2.aruco.CharucoDetector(B.make_board()).detectBoard(img)
    assert ci is not None and len(ci) == 24 and len(mi) == 17
    pos = dict(zip(ci.ravel().tolist(), cc.reshape(-1, 2)))
    spacings = [np.linalg.norm(pos[i + 1] - pos[i]) for i in pos if i % 4 != 3 and i + 1 in pos]
    spacings += [np.linalg.norm(pos[i + 4] - pos[i]) for i in pos if i + 4 in pos]
    square_mm = np.mean(spacings) / dpi * 25.4
    assert square_mm == pytest.approx(30.0, abs=0.1)
