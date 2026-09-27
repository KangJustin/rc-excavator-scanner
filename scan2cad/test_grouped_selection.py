from pathlib import Path

import cv2
import numpy as np

import select_frames


def textured_scene():
    rng = np.random.default_rng(7)
    image = np.full((480, 640, 3), 142, dtype=np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(45, 595)), int(rng.integers(35, 445))
        radius = int(rng.integers(1, 5))
        color = tuple(int(v) for v in rng.integers(35, 225, size=3))
        cv2.circle(image, (x, y), radius, color, -1)
    cv2.rectangle(image, (210, 145), (430, 335), (35, 130, 220), -1)
    cv2.rectangle(image, (210, 145), (430, 335), (20, 30, 40), 5)
    cv2.putText(image, "ROBOT SCAN", (235, 245), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (245, 245, 245), 2, cv2.LINE_AA)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(select_frames.ARUCO_DICT), 0, 70)
    patch = np.full((90, 90), 255, dtype=np.uint8)
    patch[10:80, 10:80] = marker
    image[365:455, 65:155] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return image


def transform(image, angle, dx):
    matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
    matrix[0, 2] += dx
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def main():
    root = Path("synthetic_grouped_selection")
    source = root / "input"
    output = root / "output"
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    base = textured_scene()

    poses = [(0.0, 0)] * 12
    poses += [(value * 2.5, value) for value in range(1, 13)]
    poses += [(30.0, 12)] * 12
    poses += [(30.0 + value * 2.5, 12 + value) for value in range(1, 13)]
    for index, (angle, dx) in enumerate(poses):
        cv2.imwrite(str(source / f"{index:04d}.png"), transform(base, angle, dx))

    selected_paths, summary = select_frames.select_frames(
        source, n=16, out_folder=output, groups=12, preview=True, workers=2)
    selected_items = [select_frames._analyse_one((index, str(path)))
                      for index, path in enumerate(selected_paths)]
    duplicate_pairs = []
    for i in range(len(selected_items)):
        for j in range(i + 1, len(selected_items)):
            if select_frames._near_duplicate(selected_items[i], selected_items[j]):
                duplicate_pairs.append((selected_paths[i].name, selected_paths[j].name))

    group_sizes = list(summary["frames_per_group"].values())
    checks = {
        "all_12_motion_groups_represented": not summary["groups_without_usable_pick"],
        "groups_not_equal_frame_count": max(group_sizes) - min(group_sizes) >= 8,
        "stationary_runs_collapsed": max(group_sizes) >= 10,
        "no_selected_duplicates": not duplicate_pairs,
    }
    print("Synthetic grouped summary:", summary)
    print("Selected:", [path.name for path in selected_paths])
    print("Duplicate pairs:", duplicate_pairs)
    print("Checks:", checks)
    if not all(checks.values()):
        raise SystemExit("Synthetic grouped-selection checks failed")


if __name__ == "__main__":
    main()
