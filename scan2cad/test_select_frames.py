from pathlib import Path

import cv2
import numpy as np

import select_frames


ROOT = Path("synthetic_frame_selection")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
INPUT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def sharp_scene(angle=0, shift=0):
    image = np.full((480, 640, 3), 145, dtype=np.uint8)
    center = (320 + shift, 240)
    cv2.ellipse(image, center, (135, 85), angle, 0, 360, (20, 170, 230), -1)
    cv2.ellipse(image, center, (135, 85), angle, 0, 360, (10, 40, 60), 5)
    for offset in range(-100, 101, 20):
        cv2.line(image, (center[0] + offset, 175), (center[0] + offset, 305), (40, 70, 90), 2)
    cv2.circle(image, center, 35, (220, 220, 220), -1)
    cv2.circle(image, center, 35, (20, 20, 20), 4)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(select_frames.ARUCO_DICT), 0, 72)
    patch = np.full((92, 92), 255, dtype=np.uint8)
    patch[10:82, 10:82] = marker
    image[350:442, 70:162] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return image


sharp_a = sharp_scene(0)
sharp_b = sharp_scene(25)
sharp_c = sharp_scene(-25)
blurred = cv2.GaussianBlur(sharp_a, (31, 31), 9)
exact_duplicate = sharp_a.copy()
_, near_encoded = cv2.imencode(".jpg", sharp_a, [cv2.IMWRITE_JPEG_QUALITY, 88])
near_duplicate = cv2.imdecode(near_encoded, cv2.IMREAD_COLOR)
dark = np.full_like(sharp_a, 3)
blank = np.full_like(sharp_a, 145)

fixtures = [
    ("000_sharp.png", sharp_a),
    ("001_blurred.png", blurred),
    ("002_exact_duplicate.png", exact_duplicate),
    ("003_near_duplicate.png", near_duplicate),
    ("004_sharp_view_b.png", sharp_b),
    ("005_sharp_view_c.png", sharp_c),
    ("006_dark.png", dark),
    ("007_blank.png", blank),
]
for filename, image in fixtures:
    cv2.imwrite(str(INPUT / filename), image)

def main():
    selected, summary = select_frames.select_frames(INPUT, n=3, out_folder=OUTPUT, preview=True, workers=2)
    print("Selected:", [path.name for path in selected])
    print("Summary:", summary)

    rows = {}
    with open(OUTPUT / "scores.csv", newline="", encoding="utf-8") as handle:
        import csv
        for row in csv.DictReader(handle):
            rows[row["filename"]] = row

    checks = {
        "blurred": "blur" in rows["001_blurred.png"]["rejection_reasons"],
        "exact_duplicate": rows["002_exact_duplicate.png"]["status"] == "duplicate",
        "near_duplicate_group": (rows["003_near_duplicate.png"]["status"] == "duplicate"
                                 or rows["000_sharp.png"]["status"] == "duplicate"),
        "dark": "bad_exposure" in rows["006_dark.png"]["rejection_reasons"],
        "blank": "low_detail" in rows["007_blank.png"]["rejection_reasons"],
    }
    print("Checks:", checks)
    if not all(checks.values()):
        raise SystemExit("Synthetic frame-selection checks failed")


if __name__ == "__main__":
    main()
