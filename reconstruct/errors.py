"""Reconstruction errors. Every one carries a `hint` telling you what to change."""
from __future__ import annotations


class ReconstructionError(Exception):
    def __init__(self, message: str, hint: str | None = None, log_path=None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.log_path = log_path

    def __str__(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"\n  What to try: {self.hint}")
        if self.log_path:
            parts.append(f"\n  Full log: {self.log_path}")
        return "".join(parts)


class ToolNotFoundError(ReconstructionError):
    """An external executable is missing."""


class ColmapNotFoundError(ToolNotFoundError):
    pass


class OpenMVSNotFoundError(ToolNotFoundError):
    pass


class PreflightError(ReconstructionError):
    """Problem with the input images themselves."""


class TooFewFeaturesError(ReconstructionError):
    pass


class InsufficientOverlapError(ReconstructionError):
    pass


class SparseReconstructionError(ReconstructionError):
    pass


class DegenerateReconstructionError(ReconstructionError):
    """The solved camera path can't support a 3D model (straight line / too little arc)."""


class DenseUnavailableError(ReconstructionError):
    """This dense backend cannot run here (e.g. COLMAP without CUDA)."""


class DenseReconstructionError(ReconstructionError):
    pass


class MeshingError(ReconstructionError):
    pass


class ColmapCommandError(ReconstructionError):
    """An external tool exited non-zero."""

    def __init__(self, stage: str, returncode: int, output: str, log_path=None, tool: str = "COLMAP"):
        self.stage = stage
        self.returncode = returncode
        self.output = output
        self.tool = tool
        tail = "\n".join(output.strip().splitlines()[-8:])
        super().__init__(f"{tool} `{stage}` failed (exit {returncode}).\n{tail}", log_path=log_path)
