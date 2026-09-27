#!/usr/bin/env python3
"""Select sharp, well-framed, diverse robot-scan stills without altering source images."""

import argparse
import csv
import math
import multiprocessing
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LAYOUT = {
    0: "front", 1: "right", 2: "back", 3: "left",
    4: "back-right", 5: "back-left",
}
CSV_FIELDS = [
    "filename", "status", "rejection_reasons", "duplicate_of", "group", "quality_score",
    "sharpness", "motion_change", "camera_motion", "cumulative_motion",
    "mean_brightness", "contrast", "clipped_fraction",
    "edge_density", "gradient_energy", "object_size", "center_distance", "framing_score",
    "dark_fraction", "uniform_tile_fraction", "marker_ids", "marker_count",
    "view_sides", "view_angle", "dhash",
]
DHASH_DUPLICATE_BITS = 0
ORB_DUPLICATE_RATIO = 0.97
THUMB_DUPLICATE_DISTANCE = 0.006


def _resize_for_analysis(image, width=320):
    if image.shape[1] <= width:
        return image.copy()
    scale = width / image.shape[1]
    return cv2.resize(image, (width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def _dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _marker_view(markers, height, width):
    if not markers:
        return [], "unknown"
    vectors = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0.7, 0.7), 5: (-0.7, 0.7)}
    median_size = float(np.median([m[3] for m in markers]))
    vx = vy = total = 0.0
    ys = []
    for marker_id, cx, cy, size in markers:
        weight = (size / max(median_size, 1e-6)) * (0.75 + 0.5 * cy / max(height, 1))
        dx, dy = vectors[marker_id]
        vx += dx * weight
        vy += dy * weight
        total += weight
        ys.append(cy / max(height, 1))
    vx, vy = vx / total, vy / total
    sides = []
    if vy < -0.2:
        sides.append("front")
    elif vy > 0.2:
        sides.append("back")
    if vx > 0.2:
        sides.append("right")
    elif vx < -0.2:
        sides.append("left")
    if not sides:
        nearest = max(markers, key=lambda m: (m[3], m[2]))[0]
        sides = [s for s in MARKER_LAYOUT[nearest].split("-") if s in {"front", "right", "back", "left"}]
    vertical_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    if len(markers) >= 4 or vertical_span > 0.35:
        angle = "high"
    elif float(np.mean(ys)) > 0.62 and vertical_span < 0.25:
        angle = "low"
    else:
        angle = "mid"
    return sides, angle


def _analyse_one(task):
    index, path_text = task
    cv2.setNumThreads(1)
    path = Path(path_text)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"index": index, "path": path_text, "load_error": True}
    image = _resize_for_analysis(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(lap.var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient_energy = float(gradient.mean())
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())
    clipped_fraction = float(np.mean((gray <= 2) | (gray >= 253)))
    dark_fraction = float(np.mean(gray < 35))

    rows, cols = 6, 8
    tile_scores, uniform_tiles = [], 0
    for row in range(rows):
        y0, y1 = row * h // rows, (row + 1) * h // rows
        for col in range(cols):
            x0, x1 = col * w // cols, (col + 1) * w // cols
            tile_gray = gray[y0:y1, x0:x1]
            tile_scores.append((float(gradient[y0:y1, x0:x1].mean()), row, col))
            if float(tile_gray.std()) < 7.0:
                uniform_tiles += 1
    uniform_tile_fraction = uniform_tiles / (rows * cols)
    scores_only = np.array([v for v, _, _ in tile_scores])
    active_threshold = max(float(np.percentile(scores_only, 65)), float(scores_only.mean()))
    active = [(row, col, value) for value, row, col in tile_scores if value >= active_threshold and value > 1.0]
    if active:
        weights = np.array([v for _, _, v in active])
        xs = np.array([(col + 0.5) / cols for _, col, _ in active])
        ys = np.array([(row + 0.5) / rows for row, _, _ in active])
        cx, cy = float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))
        center_distance = min(math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5), 1.0)
        min_row, max_row = min(v[0] for v in active), max(v[0] for v in active)
        min_col, max_col = min(v[1] for v in active), max(v[1] for v in active)
        object_size = ((max_row - min_row + 1) / rows) * ((max_col - min_col + 1) / cols)
        boundary = sum(row in (0, rows - 1) or col in (0, cols - 1) for row, col, _ in active) / len(active)
    else:
        center_distance, object_size, boundary = 1.0, 0.0, 1.0
    center_score = max(0.0, 1.0 - center_distance)
    size_score = math.exp(-abs(math.log(max(object_size, 0.01) / 0.38)))
    framing_score = float(np.clip(0.6 * center_score + 0.4 * size_score - 0.35 * max(boundary - 0.45, 0), 0, 1))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    markers = []
    if ids is not None:
        for marker_corners, marker_id_raw in zip(corners, ids.flatten()):
            marker_id = int(marker_id_raw)
            if marker_id not in MARKER_LAYOUT:
                continue
            points = marker_corners.reshape(4, 2)
            center = np.mean(points, axis=0)
            side = float(np.mean([np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]))
            markers.append((marker_id, float(center[0]), float(center[1]), side))
    view_sides, view_angle = _marker_view(markers, h, w)

    orb = cv2.ORB_create(nfeatures=300)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    descriptor_bytes = descriptors.tobytes() if descriptors is not None else b""
    point_array = (np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
                   if keypoints else np.empty((0, 2), dtype=np.float32))
    motion_thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).tobytes()
    return {
        "index": index, "path": path_text, "load_error": False,
        "sharpness": sharpness, "mean_brightness": mean_brightness, "contrast": contrast,
        "clipped_fraction": clipped_fraction, "edge_density": edge_density,
        "gradient_energy": gradient_energy, "object_size": object_size,
        "center_distance": center_distance, "framing_score": framing_score,
        "boundary_detail": boundary, "dark_fraction": dark_fraction,
        "uniform_tile_fraction": uniform_tile_fraction, "marker_ids": [m[0] for m in markers],
        "view_sides": view_sides, "view_angle": view_angle, "dhash": _dhash(gray),
        "orb": descriptor_bytes, "orb_rows": 0 if descriptors is None else len(descriptors),
        "orb_points": point_array.tobytes(), "orb_point_rows": len(point_array),
        "motion_thumb": motion_thumb,
    }


def _progress_map(tasks, workers):
    total = len(tasks)
    done = 0
    results = []
    width = 30
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_analyse_one, tasks, chunksize=max(1, min(32, total // max(workers * 8, 1)))):
            results.append(result)
            done += 1
            filled = round(width * done / max(total, 1))
            print(f"\rAnalysing [{'#' * filled}{'.' * (width - filled)}] {done}/{total}", end="", flush=True)
    print()
    return results


def _percentile_scale(values, value, low=10, high=90):
    lo, hi = np.percentile(values, [low, high])
    return float(np.clip((value - lo) / max(hi - lo, 1e-9), 0, 1))


def _orb_descriptors(item):
    if not item["orb_rows"]:
        return None
    return np.frombuffer(item["orb"], dtype=np.uint8).reshape(item["orb_rows"], 32)


def _orb_similarity(a, b):
    da, db = _orb_descriptors(a), _orb_descriptors(b)
    if da is None or db is None or min(len(da), len(db)) < 8:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if not matches:
        return 0.0
    good = sum(match.distance < 42 for match in matches)
    return float(good / max(min(len(da), len(db)), 1))


def _hash_distance(a, b):
    return (a["dhash"] ^ b["dhash"]).bit_count() / 64.0


def _thumb_distance(a, b):
    ta = np.frombuffer(a["motion_thumb"], dtype=np.uint8).astype(np.float32)
    tb = np.frombuffer(b["motion_thumb"], dtype=np.uint8).astype(np.float32)
    return float(np.clip(np.mean(np.abs(ta - tb)) / 35.0, 0, 1))


def _image_distance(a, b):
    hash_distance = _hash_distance(a, b)
    orb_dissimilarity = 1.0 - _orb_similarity(a, b)
    return float(np.clip(0.20 * hash_distance + 0.40 * orb_dissimilarity
                         + 0.40 * _thumb_distance(a, b), 0, 1))


def _selection_distance(a, b):
    """Viewpoint distance requested for fill selection: dHash plus ORB dissimilarity."""
    return float(np.clip(0.5 * _hash_distance(a, b) + 0.5 * (1.0 - _orb_similarity(a, b)), 0, 1))


def _near_duplicate(a, b):
    bits = (a["dhash"] ^ b["dhash"]).bit_count()
    orb = _orb_similarity(a, b) if bits <= 4 else 0.0
    hash_duplicate = bits <= DHASH_DUPLICATE_BITS and _thumb_distance(a, b) <= THUMB_DUPLICATE_DISTANCE
    return hash_duplicate or orb >= ORB_DUPLICATE_RATIO


def _camera_motion_between(previous, current):
    """Estimate normalized camera motion from a robust ORB transform, with flow/pixel fallbacks."""
    prev = np.frombuffer(previous["motion_thumb"], dtype=np.uint8).reshape(64, 64)
    curr = np.frombuffer(current["motion_thumb"], dtype=np.uint8).reshape(64, 64)
    fallback = float(cv2.absdiff(prev, curr).mean() / 255.0)
    previous_descriptors = _orb_descriptors(previous)
    current_descriptors = _orb_descriptors(current)
    if previous_descriptors is not None and current_descriptors is not None:
        try:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
                previous_descriptors, current_descriptors, k=2)
            good = [first for pair in matches if len(pair) == 2
                    for first, second in [pair] if first.distance < 0.75 * second.distance]
            if len(good) >= 8:
                previous_points = np.frombuffer(previous["orb_points"], dtype=np.float32).reshape(-1, 2)
                current_points = np.frombuffer(current["orb_points"], dtype=np.float32).reshape(-1, 2)
                src = np.float32([previous_points[match.queryIdx] for match in good])
                dst = np.float32([current_points[match.trainIdx] for match in good])
                affine, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
                if affine is not None and inliers is not None and int(inliers.sum()) >= 6:
                    translation = math.hypot(float(affine[0, 2]), float(affine[1, 2])) / math.hypot(320, 240)
                    rotation = abs(math.atan2(float(affine[1, 0]), float(affine[0, 0]))) / math.pi
                    scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
                    scale_change = abs(math.log(max(scale, 1e-6)))
                    return max(translation + 0.5 * rotation + 0.25 * scale_change, fallback * 0.15)
        except cv2.error:
            pass
    try:
        points = cv2.goodFeaturesToTrack(prev, maxCorners=120, qualityLevel=0.01, minDistance=3)
        if points is None or len(points) < 8:
            return fallback
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None,
                                                    winSize=(15, 15), maxLevel=2)
        good = status.ravel().astype(bool)
        if moved is None or np.count_nonzero(good) < 6:
            return fallback
        displacement = np.linalg.norm(moved[good] - points[good], axis=2).ravel()
        flow = float(np.median(displacement) / math.hypot(64, 64))
        return max(flow, fallback)
    except cv2.error:
        return fallback


def _assign_motion_groups(items, group_count):
    raw_motion = [0.0]
    for index in range(1, len(items)):
        if items[index - 1].get("load_error") or items[index].get("load_error"):
            motion = items[index].get("motion_change", 0.0) / 255.0
        else:
            motion = _camera_motion_between(items[index - 1], items[index])
        raw_motion.append(motion)
    positive_motion = [motion for motion in raw_motion[1:] if motion > 0]
    noise_floor = float(np.percentile(positive_motion, 25)) if positive_motion else 0.0
    adjusted_motion = [max(0.0, motion - noise_floor) for motion in raw_motion]
    stationary = sum(motion == 0.0 for motion in adjusted_motion[1:])
    print(f"Camera-motion noise floor: {noise_floor:.6f}; stationary transitions: "
          f"{stationary}/{max(len(items) - 1, 0)}")

    cumulative = 0.0
    if items:
        items[0]["camera_motion"] = 0.0
        items[0]["cumulative_motion"] = 0.0
    for index in range(1, len(items)):
        motion = adjusted_motion[index]
        cumulative += motion
        items[index]["camera_motion"] = motion
        items[index]["cumulative_motion"] = cumulative
    if not items:
        return [0.0] * (group_count + 1), [0.0] * group_count
    if cumulative <= 1e-9:
        for item in items:
            item["group"] = min(item["index"] * group_count // max(len(items), 1), group_count - 1) + 1
        boundaries = np.linspace(0, 1, group_count + 1).tolist()
    else:
        boundaries = np.linspace(0, cumulative, group_count + 1).tolist()
        inner = boundaries[1:-1]
        for item in items:
            item["group"] = min(int(np.searchsorted(inner, item["cumulative_motion"], side="right")),
                                group_count - 1) + 1
    group_motion = [0.0] * group_count
    for item in items:
        group_motion[item["group"] - 1] += item.get("camera_motion", 0.0)
    return boundaries, group_motion


def _make_groups_plot(items, selected, boundaries, output_path):
    width, height = 1200, 620
    left, right, top, bottom = 75, 35, 45, 75
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    plot_w, plot_h = width - left - right, height - top - bottom
    total = max(boundaries[-1], 1e-9)

    def point(index, cumulative):
        x = left + round(index / max(len(items) - 1, 1) * plot_w)
        y = top + plot_h - round(cumulative / total * plot_h)
        return x, y

    cv2.line(canvas, (left, top), (left, top + plot_h), (60, 60, 60), 2)
    cv2.line(canvas, (left, top + plot_h), (left + plot_w, top + plot_h), (60, 60, 60), 2)
    curve = np.array([point(item["index"], item.get("cumulative_motion", 0.0)) for item in items], np.int32)
    if len(curve) > 1:
        cv2.polylines(canvas, [curve], False, (180, 80, 20), 3, cv2.LINE_AA)
    for group_number, target in enumerate(boundaries[1:-1], 1):
        crossing = next((item for item in items if item.get("cumulative_motion", 0.0) >= target), items[-1])
        x, _ = point(crossing["index"], target)
        cv2.line(canvas, (x, top), (x, top + plot_h), (185, 185, 185), 1)
        cv2.putText(canvas, str(group_number + 1), (x + 3, top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    for item in selected:
        x, y = point(item["index"], item.get("cumulative_motion", 0.0))
        cv2.circle(canvas, (x, y), 7, (20, 30, 220), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"G{item['group']}", (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 30, 180), 1)
    cv2.putText(canvas, "Cumulative camera motion with equal-motion groups", (left, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Frame index", (width // 2 - 55, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(canvas, "motion", (8, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.imwrite(str(output_path), canvas)


def _make_contact_sheet(items, output_path, title=None, per_row=4):
    if not items:
        sheet = np.full((120, 720, 3), 245, dtype=np.uint8)
        cv2.putText(sheet, title or "Contact sheet", (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(sheet, "No matching frames.", (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (70, 70, 70), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    thumb_w, thumb_h, label_h = 300, 200, 38
    rows = math.ceil(len(items) / per_row)
    top = 36 if title else 0
    sheet = np.full((top + rows * (thumb_h + label_h), per_row * thumb_w, 3), 245, dtype=np.uint8)
    if title:
        cv2.putText(sheet, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for i, (item, label) in enumerate(items):
        image = cv2.imread(str(item["path"]))
        if image is None:
            continue
        scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        row, col = divmod(i, per_row)
        x = col * thumb_w + (thumb_w - resized.shape[1]) // 2
        y = top + row * (thumb_h + label_h) + (thumb_h - resized.shape[0]) // 2
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(sheet, label[:46], (col * thumb_w + 6, top + row * (thumb_h + label_h) + thumb_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _write_csv(items, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in sorted(items, key=lambda value: value["index"]):
            row = {field: item.get(field, "") for field in CSV_FIELDS}
            row["filename"] = Path(item["path"]).name
            row["rejection_reasons"] = ";".join(item.get("rejection_reasons", []))
            row["marker_ids"] = ";".join(map(str, item.get("marker_ids", [])))
            row["marker_count"] = len(item.get("marker_ids", []))
            row["view_sides"] = ";".join(item.get("view_sides", []))
            row["quality_score"] = f"{item.get('quality_score', 0):.5f}"
            row["dhash"] = f"{item.get('dhash', 0):016x}"
            for field in ("sharpness", "motion_change", "camera_motion", "cumulative_motion",
                          "mean_brightness", "contrast", "clipped_fraction",
                          "edge_density", "gradient_energy", "object_size", "center_distance", "framing_score",
                          "dark_fraction", "uniform_tile_fraction"):
                if field in item:
                    row[field] = f"{item[field]:.6f}"
            writer.writerow(row)


def select_frames(input_folder, n=16, out_folder=None, marker_mm=60, groups=12,
                  preview=False, workers=None):
    """Run selection and return (selected source Paths, summary dict). Source files are read-only."""
    source = Path(input_folder).resolve()
    if not source.is_dir():
        raise ValueError(f"Input folder does not exist: {source}")
    paths = sorted(path for path in source.iterdir()
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError(f"No PNG or JPEG image files found in {source}")
    output = Path(out_folder).resolve() if out_folder else source.with_name(f"{source.name}_best{n}")
    output.mkdir(parents=True, exist_ok=True)
    if output == source:
        raise ValueError("Output folder must differ from the source folder")
    previous_scores = output / "scores.csv"
    if previous_scores.exists():
        with open(previous_scores, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "selected":
                    previous_copy = output / Path(row["filename"]).name
                    if previous_copy.is_file() and previous_copy.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        previous_copy.unlink()

    worker_count = workers or max(1, min(multiprocessing.cpu_count() - 1, 12))
    print(f"Analysing {len(paths)} image frames at ~320 px with {worker_count} workers...")
    items = _progress_map([(index, str(path)) for index, path in enumerate(paths)], worker_count)
    items.sort(key=lambda value: value["index"])

    for i, item in enumerate(items):
        item["status"] = "kept"
        item["rejection_reasons"] = []
        item["duplicate_of"] = ""
        if item.get("load_error"):
            item["rejection_reasons"] = ["unreadable"]
            item["status"] = "rejected"
            continue
        changes = []
        current = np.frombuffer(item["motion_thumb"], dtype=np.uint8).reshape(64, 64)
        for neighbour_index in (i - 1, i + 1):
            if 0 <= neighbour_index < len(items) and not items[neighbour_index].get("load_error"):
                neighbour = np.frombuffer(items[neighbour_index]["motion_thumb"], dtype=np.uint8).reshape(64, 64)
                changes.append(float(cv2.absdiff(current, neighbour).mean()))
        item["motion_change"] = min(changes) if changes else 0.0

    valid = [item for item in items if not item.get("load_error")]
    sharp_values = np.array([item["sharpness"] for item in valid])
    edge_values = np.array([item["edge_density"] for item in valid])
    gradient_values = np.array([item["gradient_energy"] for item in valid])
    motion_values = np.array([item["motion_change"] for item in valid])
    sharp_min = 0.40 * float(np.median(sharp_values))
    edge_min = max(0.002, 0.28 * float(np.median(edge_values)))
    gradient_min = max(1.0, 0.28 * float(np.median(gradient_values)))
    motion_median = float(np.median(motion_values))
    motion_mad = float(np.median(np.abs(motion_values - motion_median)))
    motion_max = max(float(np.percentile(motion_values, 98)),
                     motion_median + 6.0 * max(motion_mad, 0.5))

    for item in valid:
        reasons = []
        if item["clipped_fraction"] > 0.05 or item["mean_brightness"] < 28 or item["mean_brightness"] > 228 or item["contrast"] < 12:
            reasons.append("bad_exposure")
        if item["sharpness"] < sharp_min:
            reasons.append("blur")
        if item["motion_change"] > motion_max and item["sharpness"] < 0.80 * float(np.median(sharp_values)):
            reasons.append("motion_blur")
        if item["edge_density"] < edge_min or item["gradient_energy"] < gradient_min:
            reasons.append("low_detail")
        if item["object_size"] < 0.07 or item["framing_score"] < 0.20 or (
                item["boundary_detail"] > 0.72 and item["center_distance"] > 0.38):
            reasons.append("object_missing_or_badly_framed")
        if item["dark_fraction"] > 0.55 or item["uniform_tile_fraction"] > 0.82:
            reasons.append("occlusion")
        item["rejection_reasons"] = reasons
        if reasons:
            item["status"] = "rejected"

    survivors = [item for item in valid if item["status"] == "kept"]
    score_pool = survivors or valid
    if score_pool:
        score_sharp = np.array([item["sharpness"] for item in score_pool])
        score_detail = np.array([item["gradient_energy"] for item in score_pool])
        score_contrast = np.array([item["contrast"] for item in score_pool])
        for item in valid:
            sharp_score = _percentile_scale(score_sharp, item["sharpness"])
            detail_score = _percentile_scale(score_detail, item["gradient_energy"])
            contrast_score = _percentile_scale(score_contrast, item["contrast"])
            exposure_score = float(np.clip(1.0 - abs(item["mean_brightness"] - 128) / 128
                                           - min(item["clipped_fraction"] / 0.05, 1.0) * 0.35, 0, 1))
            marker_bonus = 1.0 if item["marker_ids"] else 0.0
            item["quality_score"] = float(np.clip(
                0.27 * sharp_score + 0.18 * exposure_score + 0.18 * detail_score
                + 0.10 * contrast_score + 0.20 * item["framing_score"] + 0.07 * marker_bonus, 0, 1))

    groups = max(1, min(int(groups), max(len(items), 1)))
    boundaries, group_motion = _assign_motion_groups(items, groups)
    print(f"Duplicate guard thresholds: dHash <= {DHASH_DUPLICATE_BITS} bits with thumbnail distance "
          f"<= {THUMB_DUPLICATE_DISTANCE:.3f}, OR ORB match ratio >= {ORB_DUPLICATE_RATIO:.2f}")
    adjacent_duplicates = sum(_near_duplicate(items[index - 1], items[index])
                              for index in range(1, len(items))
                              if not items[index - 1].get("load_error") and not items[index].get("load_error"))
    print(f"Adjacent near-duplicates: {adjacent_duplicates}/{max(len(items) - 1, 0)}")

    selected = []
    group_gaps = []
    for group_number in range(1, groups + 1):
        usable = sorted((item for item in survivors if item["group"] == group_number),
                        key=lambda value: value["quality_score"], reverse=True)
        if not usable:
            group_gaps.append(group_number)
            continue
        pick = next((item for item in usable
                     if not any(_near_duplicate(item, chosen) for chosen in selected)), None)
        if pick is None:
            group_gaps.append(group_number)
            continue
        selected.append(pick)

    max_group_motion = max(group_motion) if group_motion else 0.0
    while len(selected) < min(n, len(survivors)):
        candidates = [item for item in survivors if item not in selected
                      and not any(_near_duplicate(item, chosen) for chosen in selected)]
        if not candidates:
            break
        best_item, best_value = None, -1.0
        for item in candidates:
            nearest = min((_selection_distance(item, chosen) for chosen in selected), default=1.0)
            motion_weight = (group_motion[item["group"] - 1] / max_group_motion
                             if max_group_motion > 0 else 1.0)
            value = item["quality_score"] * max(nearest, 0.01) * (0.65 + 0.35 * motion_weight)
            if value > best_value:
                best_item, best_value = item, value
        selected.append(best_item)

    duplicate_count = 0
    for item in survivors:
        if item in selected:
            continue
        duplicate = next((chosen for chosen in selected if _near_duplicate(item, chosen)), None)
        if duplicate:
            item["status"] = "duplicate"
            item["duplicate_of"] = Path(duplicate["path"]).name
            duplicate_count += 1

    for item in selected:
        item["status"] = "selected"
        shutil.copy2(item["path"], output / Path(item["path"]).name)

    _write_csv(items, output / "scores.csv")
    capture_order = sorted(selected, key=lambda value: value["index"])
    _make_contact_sheet(
        [(item, f"G{item['group']}  {Path(item['path']).name}  score={item['quality_score']:.3f}")
         for item in capture_order],
        output / "contact_sheet.jpg", title="Selected frames in capture order")
    _make_groups_plot(items, selected, boundaries, output / "groups.png")
    if preview:
        examples = []
        by_reason = defaultdict(list)
        for item in items:
            for reason in item.get("rejection_reasons", []):
                if len(by_reason[reason]) < 4:
                    by_reason[reason].append(item)
        for reason, reason_items in sorted(by_reason.items()):
            examples.extend((item, f"{reason}: {Path(item['path']).name}") for item in reason_items)
        _make_contact_sheet(examples, output / "rejected_samples.jpg", title="Rejection samples")

    reason_counts = Counter(reason for item in items for reason in item.get("rejection_reasons", []))
    covered = set()
    for item in selected:
        covered.update(item.get("view_sides", []))
        covered.add(item.get("view_angle", "unknown"))
    targets = ["front", "right", "back", "left", "high", "low"]
    missing = [target for target in targets if target not in covered]
    frames_per_group = {group_number: sum(item.get("group") == group_number for item in items)
                        for group_number in range(1, groups + 1)}
    pairwise_distances = [_selection_distance(selected[i], selected[j])
                          for i in range(len(selected)) for j in range(i + 1, len(selected))]
    minimum_pairwise_distance = min(pairwise_distances) if pairwise_distances else 1.0
    summary = {
        "total": len(items), "rejected": sum(item["status"] == "rejected" for item in items),
        "reasons": dict(reason_counts), "duplicates": duplicate_count, "selected": len(selected),
        "missing_directions": missing, "frames_per_group": frames_per_group,
        "groups_without_usable_pick": group_gaps,
        "minimum_pairwise_distance": minimum_pairwise_distance, "output": str(output),
    }
    print(f"\nSummary: {summary['total']} total, {summary['rejected']} hard-rejected, "
          f"{duplicate_count} near-duplicates removed, {len(selected)} selected")
    for reason, count in sorted(reason_counts.items()):
        print(f"  rejected by {reason}: {count}")
    print("  frames per group: " + ", ".join(
        f"G{group_number}={count}" for group_number, count in frames_per_group.items()))
    print(f"  groups without usable non-duplicate picks: {group_gaps or 'none'}")
    print(f"  minimum selected pairwise dHash/ORB distance: {minimum_pairwise_distance:.4f}")
    if missing:
        print("WARNING: selected frames have no coverage for: " + ", ".join(missing)
              + ". The scan may have missed that side/angle.")
    print(f"Outputs: {output}")
    return [Path(item["path"]) for item in capture_order], summary


def main():
    parser = argparse.ArgumentParser(description="Select the best diverse frames from robot-scan PNG stills")
    parser.add_argument("--input", required=True, help="folder containing source PNG frames (JPEG also supported)")
    parser.add_argument("--n", type=int, default=16, help="number of frames to select (default 16)")
    parser.add_argument("--out", help="output folder (default: <input>_best<N>)")
    parser.add_argument("--marker-mm", type=float, default=60, help="common ArUco marker size (default 60 mm)")
    parser.add_argument("--groups", type=int, default=12,
                        help="equal-cumulative-motion groups (default 12)")
    parser.add_argument("--preview", action="store_true", help="write rejected_samples.jpg")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.groups < 1:
        parser.error("--groups must be at least 1")
    try:
        select_frames(args.input, n=args.n, out_folder=args.out, marker_mm=args.marker_mm,
                      groups=args.groups, preview=args.preview)
    except Exception as exc:
        sys.exit(f"Frame selection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
