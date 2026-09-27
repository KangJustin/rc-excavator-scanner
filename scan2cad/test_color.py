import FreeCAD
import Part

# PARAMETERS
BOX_LENGTH = 40.0  # box length [high]
BOX_WIDTH = 30.0  # box width [high]
BOX_HEIGHT = 10.0  # box height [high]
CYLINDER_DIAMETER = 14.0  # cylinder diameter [high]
CYLINDER_HEIGHT = 18.0  # cylinder height [high]

# ASSUMPTIONS
# Test geometry has flat, unfeatured lower and rear surfaces.

# Base body: red rectangular box.
base = Part.makeBox(
    BOX_LENGTH,
    BOX_WIDTH,
    BOX_HEIGHT,
    FreeCAD.Vector(-BOX_LENGTH / 2, -BOX_WIDTH / 2, 0),
)

# Added feature: blue cylinder centered on top of the base.
cylinder = Part.makeCylinder(
    CYLINDER_DIAMETER / 2,
    CYLINDER_HEIGHT,
    FreeCAD.Vector(0, 0, BOX_HEIGHT),
)

COLORS = {"base": "#C83B32", "cylinder": "#2F65C7"}
PARTS = {"base": base, "cylinder": cylinder}
result = base.fuse(cylinder)
