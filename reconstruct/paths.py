"""'latest' as a folder name: the newest session, so nobody has to type (or mistype) a timestamp."""
from __future__ import annotations

from pathlib import Path


def resolve_latest(arg: Path, root: str, must_contain: tuple[str, ...]) -> Path:
    """If `arg` is the word 'latest', return the newest folder under `root` that has one of the `must_contain`
    file patterns in it. Anything else is returned unchanged."""
    if str(arg) != "latest":
        return arg
    base = Path(root)
    candidates = [d for d in base.glob("*") if d.is_dir() and any(any(d.glob(pat)) for pat in must_contain)] if base.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"'latest' needs a folder under ./{root}/ containing {' or '.join(must_contain)} "
                                f"(nothing found; run from the project folder).")
    newest = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"latest -> {newest}")
    return newest
