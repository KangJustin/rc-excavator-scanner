"""A ChArUco scale board with EXACT physical dimensions, as a printable vector PDF.

Why: photogrammetry has no scale. A printed board of known size in the scene lets software
work out millimetres-per-model-unit. Everything about the board's true size lives in ONE place
(BOARD, below); the PDF and charuco_board.json are both generated from it, so what is printed
and what the software believes can't drift apart.

    python -m measure.board            # writes scale_board/charuco_{letter,a4}.pdf + charuco_board.json + preview

Print at 100% / "Actual size" (NOT "fit to page"), then check the 100 mm ruler printed on the sheet.

The board is drawn as vector rectangles (no raster resampling), and all black areas are one filled
path so adjacent rectangles never show hairline seams.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- the single source of truth for the board's size ------------------------------------------
BOARD = {
    "squares_x": 5,
    "squares_y": 7,
    "square_mm": 30.0,  # side of one chessboard square, as printed
    "marker_mm": 22.5,  # side of one ArUco tag (inside its white square)
    "dictionary": "DICT_4X4_50",
}
PX_PER_SQUARE = 64  # render grid: 64 px/square, 48 px/tag = 8 px per tag cell -> every edge on a pixel
TOP_MARGIN_MM = 20.0
PAPERS_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}  # (width, height)
PT_PER_MM = 72.0 / 25.4


def board_size_mm() -> tuple[float, float]:
    return BOARD["squares_x"] * BOARD["square_mm"], BOARD["squares_y"] * BOARD["square_mm"]


def make_board():
    """The OpenCV CharucoBoard matching BOARD (lengths in mm)."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD["dictionary"]))
    return cv2.aruco.CharucoBoard((BOARD["squares_x"], BOARD["squares_y"]),
                                  BOARD["square_mm"], BOARD["marker_mm"], d)


def render_pixels() -> np.ndarray:
    """The board as a pixel-exact bitmap (0 = black, 255 = white), PX_PER_SQUARE px per square."""
    board = make_board()
    size = (BOARD["squares_x"] * PX_PER_SQUARE, BOARD["squares_y"] * PX_PER_SQUARE)
    return board.generateImage(size, marginSize=0, borderBits=1)


def black_rects(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy maximal rectangles (x, y, w, h) in pixel units covering exactly the black pixels."""
    black = img == 0
    done = np.zeros_like(black)
    rects = []
    H, W = black.shape
    for y in range(H):
        x = 0
        while x < W:
            if not black[y, x] or done[y, x]:
                x += 1
                continue
            w = 1
            while x + w < W and black[y, x + w] and not done[y, x + w]:
                w += 1
            h = 1
            while y + h < H and black[y + h, x:x + w].all() and not done[y + h, x:x + w].any():
                h += 1
            done[y:y + h, x:x + w] = True
            rects.append((x, y, w, h))
            x += w
    return rects


def rasterize_rects(rects, shape) -> np.ndarray:
    """Inverse of black_rects (used by tests to prove the rectangles equal the bitmap)."""
    img = np.full(shape, 255, np.uint8)
    for x, y, w, h in rects:
        img[y:y + h, x:x + w] = 0
    return img


# ---- minimal PDF writer (vector rectangles + Helvetica text; no dependencies) --------------------
def _pdf(page_w_mm, page_h_mm, content: str) -> bytes:
    W, H = page_w_mm * PT_PER_MM, page_h_mm * PT_PER_MM
    stream = content.encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.3f} {H:.3f}] /Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _mm(v: float) -> str:
    return f"{v * PT_PER_MM:.4f}"


def build_pdf(paper: str) -> bytes:
    page_w, page_h = PAPERS_MM[paper]
    bw, bh = board_size_mm()
    x0 = (page_w - bw) / 2
    y_top = TOP_MARGIN_MM
    px_mm = BOARD["square_mm"] / PX_PER_SQUARE

    def rect(x_mm, ytop_mm, w_mm, h_mm):  # PDF origin is bottom-left; ours is top-left
        return f"{_mm(x_mm)} {_mm(page_h - ytop_mm - h_mm)} {_mm(w_mm)} {_mm(h_mm)} re\n"

    c = ["0 g\n"]  # black fill
    # 1) the board: every black rectangle in ONE path, filled once (no seams)
    for (x, y, w, h) in black_rects(render_pixels()):
        c.append(rect(x0 + x * px_mm, y_top + y * px_mm, w * px_mm, h * px_mm))
    c.append("f\n")

    def text(x_mm, ytop_mm, size, s, bold=False):
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /{'F2' if bold else 'F1'} {size} Tf {_mm(x_mm)} {_mm(page_h - ytop_mm)} Td ({s}) Tj ET\n"

    # 2) verification ruler: exactly 100 mm, ticks every 10 mm
    ry = y_top + bh + 12
    rx = (page_w - 100.0) / 2
    c.append(rect(rx, ry, 100.0, 0.3))
    for i in range(0, 11):
        c.append(rect(rx + i * 10 - 0.15, ry - (4 if i in (0, 10) else 2.5), 0.3, (4 if i in (0, 10) else 2.5) + 0.3))
    c.append("f\n")
    c.append(text(rx - 0.5, ry - 6, 8, "0"))
    c.append(text(rx + 100 - 8, ry - 6, 8, "100 mm"))
    # 3) instructions
    c.append(text(x0, ry + 9, 10,
                  "PRINT AT 100% / ACTUAL SIZE (not 'fit to page'). Then this line must measure 100 mm.", True))
    c.append(text(x0, ry + 15, 9,
                  f"ChArUco {BOARD['squares_x']}x{BOARD['squares_y']}, square {BOARD['square_mm']:g} mm, "
                  f"tag {BOARD['marker_mm']:g} mm, {BOARD['dictionary']}. Keep flat; do not crop or edit."))
    return _pdf(page_w, page_h, "".join(c))


def spec_dict() -> dict:
    import cv2
    bw, bh = board_size_mm()
    return {
        **BOARD,
        "board_width_mm": bw, "board_height_mm": bh,
        "inner_corners": (BOARD["squares_x"] - 1) * (BOARD["squares_y"] - 1),
        "opencv_version": cv2.__version__, "legacy_pattern": False,
        "printed_square_mm_measured": None,
        "note": "After printing, measure one square with a ruler. If it is not 30.0 mm, put the real value in "
                "printed_square_mm_measured; software should prefer it over square_mm.",
    }


def write_all(out_dir: Path) -> list[Path]:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for paper in PAPERS_MM:
        p = out_dir / f"charuco_{paper}.pdf"
        p.write_bytes(build_pdf(paper))
        files.append(p)
    spec = out_dir / "charuco_board.json"
    spec.write_text(json.dumps(spec_dict(), indent=2))
    prev = out_dir / "charuco_preview.png"
    img = cv2.copyMakeBorder(cv2.resize(render_pixels(), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                             40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(prev), img)
    return files + [spec, prev]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.board", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("scale_board"))
    args = p.parse_args(argv)
    files = write_all(args.out)
    bw, bh = board_size_mm()
    print(f"ChArUco board {BOARD['squares_x']}x{BOARD['squares_y']}, {BOARD['square_mm']:g} mm squares "
          f"= {bw:g} x {bh:g} mm")
    for f in files:
        print(f"  {f}")
    print("\nPrint charuco_letter.pdf (US Letter) or charuco_a4.pdf at 100% / Actual size, then verify the "
          "100 mm ruler on the sheet with a real ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
