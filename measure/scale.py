"""Scale from a size you already know (a can's diameter, a ball's diameter), so nobody has to divide by hand.

The scan gives the object's dimensions in arbitrary model units. If you know one real dimension, the ratio is
the scale. If you know two, one sets the scale and the other becomes an independent accuracy check.
"""
from __future__ import annotations

from .analyze import MeasureError


MAX_CHECK_ERROR_PERCENT = 25.0


def _guard(info: dict, allow_mismatch: bool) -> None:
    """The independent check exists to catch exactly this: the wrong thing was isolated, or the sizes are for a
    different object. Printing the number is not enough, so stop unless the user overrides."""
    if allow_mismatch:
        return
    for k, c in info.get("independent_check", {}).items():
        if abs(c["error_percent"]) > MAX_CHECK_ERROR_PERCENT:
            raise MeasureError(
                f"The two sizes you gave disagree with the scan: the {k} came out {c['measured_mm']:.1f} mm but you said "
                f"{c['expected_mm']:g} mm ({c['error_percent']:+.0f}%). Something is wrong.",
                hint="Open preview.png: is the object actually there (a hole means it failed to reconstruct: shiny, "
                     "transparent or plain objects often do), is it the object these sizes describe, and did the tool "
                     "isolate it? Fix that and rerun, or pass --allow-mismatch to continue anyway.")


def scale_from_known(unscaled: dict, diameter_mm: float | None = None, height_mm: float | None = None,
                     allow_mismatch: bool = False):
    """unscaled = measure.analyze() result computed WITHOUT a scale (values are in model units).
    Returns (mm_per_unit, info) where info records how it was derived and any check against a second size."""
    if diameter_mm is None and height_mm is None:
        raise MeasureError("Give --known-diameter-mm and/or --known-height-mm.")
    pf = unscaled["primitive_fit"]
    shape = pf["likely_shape"].split()[0]
    if shape == "flat":
        raise MeasureError(
            "No object found: the reconstruction contains only a flat surface (a floor or table).",
            hint="The object probably failed to reconstruct (a hole in the mesh: shiny, transparent or plain objects often "
                 "do). Check preview.png and reshoot with a matte or textured object.")
    if shape not in ("cylinder", "sphere"):
        return _scale_from_standing_height(unscaled, pf, diameter_mm, height_mm, allow_mismatch)
    fit = pf[shape]
    measured = {"diameter": fit["diameter_mm"]}  # in model units, because the analysis was unscaled
    if shape == "cylinder":
        measured["height"] = fit["height_mm"]
    known = {"diameter": diameter_mm, "height": height_mm}
    if known["height"] is not None and "height" not in measured:
        raise MeasureError("A sphere has no separate height; use --known-diameter-mm.")

    primary = "diameter" if diameter_mm is not None else "height"
    scale = known[primary] / measured[primary]
    info = {"source": f"known {primary} = {known[primary]:g} mm", "shape": shape, "fit_quality": pf["fit_quality"],
            "scan_size_units": {k: round(v, 5) for k, v in measured.items()}}
    checks = {}
    for k in ("diameter", "height"):
        if k != primary and known[k] is not None:
            got = measured[k] * scale
            checks[k] = {"expected_mm": known[k], "measured_mm": round(got, 2),
                         "error_percent": round(100 * (got - known[k]) / known[k], 2)}
    if checks:
        info["independent_check"] = checks
    _guard(info, allow_mismatch)
    return scale, info


def _scale_from_standing_height(unscaled: dict, pf: dict, diameter_mm, height_mm, allow_mismatch=False):
    """The shape fit is poor (a lumpy scan of a can, say), so a fitted diameter can't be trusted. But if the object
    stands on a table that was found, its height above the table is well defined whatever its shape."""
    plane_h = unscaled["bounding_box"].get("height_above_base_plane_mm")
    if height_mm is None or not plane_h:
        raise MeasureError(
            f"Scaling from a known diameter needs the object to be a cylinder or sphere; the scan looks like "
            f"'{pf['likely_shape']}'.",
            hint="Give --known-height-mm as well (the object's standing height is measured even when the shape fit "
                 "is poor), or use --scale-mm (real size / the scan's size).")
    scale = height_mm / plane_h
    info = {"source": f"known height = {height_mm:g} mm (standing height above the table; the shape fit was "
                      f"{pf['fit_quality']}, so this is less exact)",
            "shape": pf["likely_shape"].split()[0], "fit_quality": pf["fit_quality"],
            "scan_size_units": {"height_above_table": round(plane_h, 5)}}
    if diameter_mm is not None:  # independent check: the two shorter sides of the oriented bounding box
        o = unscaled["bounding_box"]["oriented_mm"]
        width = (o["width"] + o["height"]) / 2 * scale
        info["independent_check"] = {"width": {
            "expected_mm": diameter_mm, "measured_mm": round(width, 2),
            "error_percent": round(100 * (width - diameter_mm) / diameter_mm, 2),
            "note": "bounding-box width, not a fitted diameter (the fit was poor)"}}
    _guard(info, allow_mismatch)
    return scale, info
