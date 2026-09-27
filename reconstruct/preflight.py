"""Cheap checks on the input images, before spending minutes in COLMAP."""
from __future__ import annotations

import logging
from pathlib import Path

from .errors import PreflightError

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

MIN_IMAGES_FATAL = 3
MIN_IMAGES_RECOMMENDED = 15
MIN_BRIGHTNESS = 90  # median mean-grey of the photos, out of 255


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def preflight(image_dir: Path) -> list[Path]:
    """Validate the image folder; returns the image list. Warnings are logged,
    unusable input raises PreflightError."""
    if not image_dir.is_dir():
        raise PreflightError(f"Image folder not found: {image_dir}")
    images = list_images(image_dir)
    n = len(images)
    if n < MIN_IMAGES_FATAL:
        raise PreflightError(
            f"Only {n} image(s) in {image_dir}; need at least {MIN_IMAGES_FATAL}.",
            hint="Capture 15-25 views around the object, ~15-25 degrees apart.",
        )
    if n < MIN_IMAGES_RECOMMENDED:
        log.warning("Only %d images; 15-25 spaced evenly around the object works far better.", n)

    try:
        import cv2
    except ImportError:
        log.warning("OpenCV not installed; skipping image size/blur checks.")
        return images

    sizes, sharp, unreadable, bright = {}, {}, [], {}
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            unreadable.append(p.name)
            continue
        sizes[p.name] = img.shape[::-1]
        sharp[p.name] = float(cv2.Laplacian(img, cv2.CV_64F).var())
        bright[p.name] = float(img.mean())

    if unreadable:
        raise PreflightError(f"Unreadable image file(s): {', '.join(unreadable)}",
                             hint="Delete or re-capture them.")
    if len(set(sizes.values())) > 1:
        raise PreflightError(
            "Images have different resolutions: "
            + ", ".join(f"{s[0]}x{s[1]}" for s in sorted(set(sizes.values()))),
            hint="COLMAP is run with one shared camera model, so every image must come "
                 "from the same camera at the same resolution. Remove the odd ones out.",
        )

    med_b = sorted(bright.values())[len(bright) // 2]
    if med_b < MIN_BRIGHTNESS:
        log.warning("The photos are dark (median brightness %.0f/255, want at least %d). Dark frames are noisy, match "
                    "poorly and make the colours in the model much darker than the real object. Add light "
                    "(diffuse, from several sides) and reshoot if the result is poor.", med_b, MIN_BRIGHTNESS)
    med = sorted(sharp.values())[len(sharp) // 2]
    blurry = sorted(n for n, s in sharp.items() if s < 0.4 * med)
    if blurry:
        log.warning("%d image(s) are much blurrier than the rest (likely motion blur; "
                    "consider re-shooting or deleting): %s", len(blurry), ", ".join(blurry))
    return images
