"""Give a headlessly-built .FCStd the 3D-view state it can't record itself.

FreeCAD's headless mode has no viewport, so the files it saves have no GuiDocument.xml. Opened in the
GUI, such a file shows a PartDesign Body hidden and its tip feature hidden: an empty 3D view. This adds
the missing GuiDocument.xml: the result visible, construction geometry (sketch, planes) hidden, and an
isometric camera fitted to the model.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

_PROVIDER = (
    '        <ViewProvider name="{name}" expanded="0">\n'
    '            <Properties Count="{count}" TransientCount="0">\n'
    '                <Property name="Visibility" type="App::PropertyBool" status="1">\n'
    '                    <Bool value="{vis}"/>\n'
    "                </Property>\n"
    "{extra}"
    "            </Properties>\n"
    "        </ViewProvider>\n"
)
_APPEARANCE = (
    '                <Property name="ShapeAppearance" type="App::PropertyMaterialList" status="1">\n'
    '                    <MaterialList file="{file}" version="3"/>\n'
    "                </Property>\n"
)


def appearance_blob(rgb) -> bytes:
    """FreeCAD's binary ShapeAppearance (one material): count, then ambient / diffuse / specular / emissive
    as [alpha, B, G, R] (blue first!), then shininess (float) and padding. Layout decoded from files FreeCAD
    itself saved, and the channel order confirmed by opening a test file in the FreeCAD GUI (an orange model
    came out blue when written as R, G, B)."""
    import struct
    r, g, b = (int(v) for v in rgb)
    return (struct.pack("<I", 1) + bytes([255, 0x55, 0x55, 0x55]) + bytes([255, b, g, r]) + bytes([255, 0x88, 0x88, 0x88])
            + bytes([255, 0, 0, 0]) + struct.pack("<f", 0.9) + b"\x00" * 16)


def _objects(document_xml: str) -> list[tuple[str, str]]:
    """[(type, name)] in document order."""
    return re.findall(r'<Object type="([^"]+)"[^>]*?name="([^"]+)"', document_xml.split("<ObjectData", 1)[0])


def gui_document_xml(objects: list[tuple[str, str]], bbox_mm, color=None) -> tuple[str, dict]:
    """objects: [(type, name)]; bbox_mm: (x, y, z) sizes of the model, which stands on Z=0 centred on X/Y;
    color: optional (r, g, b) 0-255 for the visible result. Returns (xml, {extra zip file name: bytes})."""
    names = [n for _, n in objects]
    visible = set()
    parts = [n for t, n in objects if t.startswith("PartDesign::") and t != "PartDesign::Body"]
    if any(t == "PartDesign::Body" for t, _ in objects):
        visible.add(next(n for t, n in objects if t == "PartDesign::Body"))
        if parts:
            visible.add(parts[-1])  # a Body draws its tip feature; if that is hidden the view is empty
    else:
        visible.update(n for t, n in objects if t.startswith("Part::"))
    files, chunks = {}, []
    for n in names:
        extra = ""
        if color is not None and n in visible:
            files[f"ShapeAppearance_{n}"] = appearance_blob(color)
            extra = _APPEARANCE.format(file=f"ShapeAppearance_{n}")
        chunks.append(_PROVIDER.format(name=n, vis="true" if n in visible else "false", extra=extra,
                                       count=2 if extra else 1))
    body = "".join(chunks)

    x, y, z = (float(v) for v in bbox_mm)
    d = (x * x + y * y + z * z) ** 0.5 or 1.0
    cx, cy, cz = 0.0, 0.0, z / 2  # model centre
    k = d / 2 / 3 ** 0.5  # isometric viewing direction (1, -1, 1)
    cam = ("OrthographicCamera {&#10;  viewportMapping ADJUST_CAMERA&#10;"
           f"  position {cx + k:.6f} {cy - k:.6f} {cz + k:.6f}&#10;"
           "  orientation 0.74290609 0.30772209 0.59447283  1.2171158&#10;"
           f"  nearDistance {d * 1e-4:.6e}&#10;  farDistance {d:.6f}&#10;  aspectRatio 1&#10;"
           f"  focalDistance {d / 2:.6f}&#10;  height {d * 1.2:.6f}&#10;&#10;}}&#10;")
    expand = "".join(f'        <Expand name="{n}"/>\n' for t, n in objects if t == "PartDesign::Body")
    xml = ("<?xml version='1.0' encoding='utf-8'?>\n"
            '<Document SchemaVersion="1" HasExpansion="1">\n'
            + (f'    <Expand count="{expand.count("<Expand")}">\n{expand}    </Expand>\n' if expand else "")
            + f'    <ViewProviderData Count="{len(names)}">\n{body}    </ViewProviderData>\n'
            + f'    <Camera settings="{cam}"/>\n</Document>\n')
    return xml, files


def add_gui_state(fcstd: Path, bbox_mm, color=None) -> bool:
    """Add GuiDocument.xml (and the colour, if given) to the file in place. Returns False (and does nothing)
    if it already has one."""
    fcstd = Path(fcstd)
    with zipfile.ZipFile(fcstd) as z:
        if "GuiDocument.xml" in z.namelist():
            return False
        doc = z.read("Document.xml").decode("utf-8")
        entries = [(i, z.read(i.filename)) for i in z.infolist()]
    xml, files = gui_document_xml(_objects(doc), bbox_mm, color)
    tmp = fcstd.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info, data in entries:
            out.writestr(info, data)
        out.writestr("GuiDocument.xml", xml)
        for name, data in files.items():
            out.writestr(name, data)
    tmp.replace(fcstd)
    return True
