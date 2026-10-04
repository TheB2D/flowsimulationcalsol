"""
STEP loading, body classification, inventory and mesh cache for the CalSol
Excalibur battery box (`BoxAssembly.step`).

The STEP is an AP242 export from Onshape.  It is read with the OCAF/XCAF
document reader rather than the plain `importStep` helper, because XCAF
preserves the assembly tree: every solid keeps the *name* of the part it came
from and the colour Onshape assigned it.  Classifying on names is far more
robust than guessing from bounding boxes, and the names are what tie the
geometry back to the CAD BOM.

Units: the file declares millimetres, and the check in `verify_units` confirms
it against a known 18650 cell (18 mm diameter x 65 mm long).

Everything expensive -- reading, exploding the assembly, tessellating ~2800
solids -- is cached to disk, so only the first run pays for it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import pathlib
import re
import pickle
import time
from dataclasses import dataclass, asdict

import numpy as np

from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.Quantity import Quantity_Color
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder


ROOT = pathlib.Path(__file__).resolve().parent.parent
STEP_PATH = ROOT / "BoxAssembly.step"
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"

# The "full battery box" CAD set.  `Battery Box.step` is the only one of the
# four exported with true B-rep solids (the rest are tessellated meshes), which
# is why the enclosure and its cutouts are measured from it.
FULLBOX_DIR = ROOT / "fullbatterybox"
MAIN_BATTERY_STEP = FULLBOX_DIR / "Main Battery.step"   # box + pack + electronics
BATTERY_BOX_STEP = FULLBOX_DIR / "Battery Box.step"     # enclosure alone
BATTERY_PACK_STEP = FULLBOX_DIR / "Battery Pack.step"   # the 17-module pack

GEOMETRY_SOURCES = {
    "legacy": STEP_PATH,          #  2804 solids, the original simple box
    "box": BATTERY_BOX_STEP,      #    90 solids, ~2.4 s -- the app default
    "full": MAIN_BATTERY_STEP,    #  3374 solids, ~45 s cold
    "pack": BATTERY_PACK_STEP,
}


def resolve_source(key_or_path) -> pathlib.Path:
    """Accept 'legacy' | 'box' | 'full' | 'pack', or an explicit path."""
    if key_or_path is None:
        return STEP_PATH
    if isinstance(key_or_path, str) and key_or_path in GEOMETRY_SOURCES:
        return GEOMETRY_SOURCES[key_or_path]
    return pathlib.Path(key_or_path)


# -- classification thresholds ------------------------------------------------
# An LG INR18650 MJ1: nominal 18.4 mm diameter, 65.0 mm long.
CELL_DIAMETER_MM = 18.4
CELL_LENGTH_MM = 65.0
CELL_TOL = 1.5          # mm, how far a bounding box may stray and still be a cell

# A case plate is large in two dimensions and thin in the third.
PLATE_MIN_SPAN_MM = 150.0
PLATE_MAX_THICKNESS_MM = 25.0

# -- the new enclosure, by CAD part name --------------------------------------
# Names are authoritative and colour is NOT: OCCT's XCAF returns *linear* RGB,
# so the blue front panel reads back as #000BA4 rather than the #023DD2 stored
# in the STEP.  Matching on colour would silently fail, so every panel here is
# matched on its name.
CASE_PANEL_NAMES = {
    "front plate", "rear plate", "side plate", "base plate",
    "top brace", "rear panel", "front duct",
    "extra piece for rear plate", "battery box",
}

# Parts the user wants transparent by default and independently toggleable.
# These used to fall into `case_plate` via the size fallback -- `Shelf Panel-1`
# (274 x 8.2 x 454 mm) and the BMS PCB (158 x 1.6 x 249 mm) both trip the
# "big and thin" test -- which welded them to the enclosure and made the
# transparency toggle unable to separate them.
SHELF_NAMES = {"electronics shelf", "shelf panel"}
PCB_NAME_HINTS = ("mainbms", "hv_pdb", "pcb", "modulelogic")

_INSTANCE_RE = re.compile(r"\s*<\d+>\s*$")
_TRAILING_IDX_RE = re.compile(r"-\d+$")


def normalise_part_name(name: str) -> str:
    """'Front Plate-1 <2>' -> 'front plate'.

    Strips Onshape's instance decorations so a part can be recognised by name
    regardless of which occurrence it is.
    """
    n = name.lower().split("/")[-1]
    n = n.split("__")[0]
    n = _INSTANCE_RE.sub("", n).strip()
    n = _TRAILING_IDX_RE.sub("", n).strip()
    return n


# =============================================================================
# Inventory record
# =============================================================================

@dataclass
class Body:
    """One solid from the STEP, with everything the flow model needs."""

    index: int
    name: str                  # part name from the STEP assembly tree
    cls: str                   # 'cell' | 'case_plate' | 'busbar' | 'structure' | 'fastener' | 'other'
    volume_mm3: float
    cx: float                  # centroid, mm
    cy: float
    cz: float
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    color: tuple[float, float, float] | None = None
    radius_mm: float | None = None      # dominant cylindrical face, if any

    @property
    def dx(self) -> float:
        return self.xmax - self.xmin

    @property
    def dy(self) -> float:
        return self.ymax - self.ymin

    @property
    def dz(self) -> float:
        return self.zmax - self.zmin

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz])


# =============================================================================
# STEP reading (XCAF -- keeps names and colours)
# =============================================================================

def _label_name(label: TDF_Label) -> str:
    """Read the TDataStd_Name attribute off an OCAF label, if present."""
    from OCP.TDataStd import TDataStd_Name as _N
    attr = _N()
    if label.FindAttribute(_N.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def read_step(path: pathlib.Path = STEP_PATH, verbose: bool = True):
    """Read the STEP into an XCAF document.

    Returns (shape_tool, color_tool, doc).  The document must be kept alive for
    as long as the tools are used, hence returning it.
    """
    app = XCAFApp_Application.GetApplication_s()
    fmt = TCollection_ExtendedString("MDTV-XCAF")
    doc = TDocStd_Document(fmt)
    app.NewDocument(fmt, doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)

    t0 = time.time()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for {path} (status {status})")
    reader.Transfer(doc)
    if verbose:
        print(f"  read+transfer: {time.time() - t0:.1f}s")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    return shape_tool, color_tool, doc


def _iter_leaf_shapes(shape_tool, color_tool, label: TDF_Label, loc: TopLoc_Location,
                      name_stack: list[str], out: list):
    """Walk the XCAF assembly tree, accumulating placed leaf shapes.

    An assembly node's children are *references*; each reference carries its own
    location, and points at a prototype label whose geometry is shared between
    all instances.  Composing the locations down the tree is what turns the 70
    unique part bodies in the file into the ~2800 placed solids of the assembly.
    """
    name = _label_name(label)

    if shape_tool.IsAssembly_s(label):
        children = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, children)
        for i in range(1, children.Length() + 1):
            child = children.Value(i)
            child_loc = shape_tool.GetLocation_s(child)
            combined = loc.Multiplied(child_loc)

            if shape_tool.IsReference_s(child):
                ref = TDF_Label()
                shape_tool.GetReferredShape_s(child, ref)
                # the instance name lives on the reference, the geometry on the
                # prototype -- prefer the instance name when it exists
                inst_name = _label_name(child) or _label_name(ref)
                _iter_leaf_shapes(shape_tool, color_tool, ref, combined,
                                  name_stack + [inst_name], out)
            else:
                _iter_leaf_shapes(shape_tool, color_tool, child, combined,
                                  name_stack + [_label_name(child)], out)
        return

    # leaf: a simple shape
    shape = shape_tool.GetShape_s(label)
    if shape.IsNull():
        return

    col = Quantity_Color()
    has_color = (color_tool.GetColor_s(label, XCAFDoc_ColorSurf, col)
                 or color_tool.GetColor_s(label, XCAFDoc_ColorGen, col))
    rgb = (col.Red(), col.Green(), col.Blue()) if has_color else None

    leaf_name = name_stack[-1] if name_stack else name
    out.append((shape, loc, leaf_name, rgb))


def explode_assembly(shape_tool, color_tool, verbose: bool = True) -> list:
    """Return [(TopoDS_Solid, name, rgb), ...] for every placed solid."""
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    if verbose:
        print(f"  root labels: {roots.Length()}")

    placed: list = []
    for i in range(1, roots.Length() + 1):
        root = roots.Value(i)
        _iter_leaf_shapes(shape_tool, color_tool, root,
                          shape_tool.GetLocation_s(root), [_label_name(root)], placed)

    if verbose:
        print(f"  placed leaf shapes: {len(placed)}")

    solids: list = []
    for shape, loc, name, rgb in placed:
        moved = shape.Moved(loc) if not loc.IsIdentity() else shape
        exp = TopExp_Explorer(moved, TopAbs_SOLID)
        while exp.More():
            solids.append((TopoDS.Solid_s(exp.Current()), name, rgb))
            exp.Next()

    if verbose:
        print(f"  solids: {len(solids)}")
    return solids


# =============================================================================
# Classification
# =============================================================================

def _sorted_extents(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    return tuple(sorted((dx, dy, dz)))  # type: ignore[return-value]


def classify(name: str, dx: float, dy: float, dz: float, volume: float) -> str:
    """Assign a body to a class.

    Name first -- the STEP carries the CAD part names, which are authoritative.
    Geometry is the fallback for anything unnamed, and the cross-check that the
    name-based answer is sane.
    """
    n = name.lower()
    key = normalise_part_name(name)
    small, mid, large = _sorted_extents(dx, dy, dz)

    # -- cells: name it, then confirm the shape really is an 18650 -----------
    if "18650" in n or "inr18650" in n:
        return "cell"
    # unnamed fallback: a cylinder 18 mm across and 65 mm long, i.e. two equal
    # small extents (the diameter) and one long one
    if (abs(large - CELL_LENGTH_MM) < CELL_TOL
            and abs(small - CELL_DIAMETER_MM) < CELL_TOL
            and abs(mid - CELL_DIAMETER_MM) < CELL_TOL):
        return "cell"

    # -- enclosure panels, by exact CAD part name ---------------------------
    # The new box names its panels "Front Plate", "Rear Plate" and so on, which
    # the old "8-000N Box_..." prefix rule never matched.  Name first, because
    # the size fallback below is both over- and under-inclusive: it would also
    # swallow the shelf panel and the BMS board, while missing `Front Duct`
    # (34 mm thick) and `Top Brace` (30 mm mid-span).
    if key in CASE_PANEL_NAMES:
        return "case_plate"
    # the original scheme, kept so BoxAssembly.step still classifies
    if n.startswith("8-000") and "box" in n:
        return "case_plate"

    # -- the parts that get their own visibility group ----------------------
    if key in SHELF_NAMES:
        return "shelf"
    # Standoffs and their tape carry "PCB" in the name but are hardware, not
    # boards -- test them before the PCB hints so they stay `structure`.
    if "standoff" in n:
        return "structure"
    if any(h in n for h in PCB_NAME_HINTS):
        return "pcb"

    # -- busbars BEFORE the divider rule ------------------------------------
    # "Divider Board Busbar-N" is a busbar.  The old ordering tested "divider"
    # first and sent it to `structure`, where it would be drawn at 10% opacity
    # and decimated away.
    if "busbar" in n or "bus bar" in n or "interconnect" in n or "terminal" in n \
            or "conductor" in n or "crossbar" in n or "tab " in n or "v_r" in n:
        return "busbar"

    if "divider" in n:
        return "structure"

    if "spacer" in n or "tie rod" in n or "rod" in n \
            or "mount" in n or "insert" in n or "tape" in n or "cover" in n \
            or "insulation" in n or "sheet" in n or "standoff" in n \
            or "fillet" in n or "magnet" in n:
        return "structure"

    if "screw" in n or "washer" in n or "rivet" in n or "nut" in n or "bolt" in n:
        return "fastener"

    # -- size fallback, LAST and deliberately narrow ------------------------
    # Only for genuinely unnamed bodies.  Every named panel was caught above,
    # so this can no longer misfile the shelf or a PCB as enclosure.
    if (not key
            and large > PLATE_MIN_SPAN_MM and mid > PLATE_MIN_SPAN_MM
            and small < PLATE_MAX_THICKNESS_MM):
        return "case_plate"

    return "other"


def named_parts(bodies: list["Body"], min_volume_mm3: float = 1000.0
                ) -> dict[str, list[int]]:
    """{normalised part name: [body indices]}, largest total volume first.

    Discovered from the inventory rather than hard-coded, because the part
    roster differs per STEP: `Electronics Shelf` is a PRODUCT name in
    `Battery Box.step` but only carries solids in `Main Battery.step`.

    `min_volume_mm3` keeps the UI list to the parts worth toggling instead of
    two hundred washers.
    """
    groups: dict[str, list[int]] = {}
    volume: dict[str, float] = {}
    for b in bodies:
        key = normalise_part_name(b.name)
        if not key:
            continue
        groups.setdefault(key, []).append(b.index)
        volume[key] = volume.get(key, 0.0) + b.volume_mm3
    keep = {k: v for k, v in groups.items() if volume[k] >= min_volume_mm3}
    return dict(sorted(keep.items(), key=lambda kv: -volume[kv[0]]))


# =============================================================================
# Inventory
# =============================================================================

def cylinder_radius(solid) -> float | None:
    """Radius of the dominant cylindrical face of a solid, in mm.

    A cell's bounding box is not its diameter: the terminal button, the tab
    and the heat-shrink wrap push the box out to ~19.9 mm on a can that is
    actually 18.4 mm.  The flow model needs the can, because that is the
    cylinder the air sees, so measure the cylindrical face directly.
    """
    counts: dict[float, int] = {}
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        ad = BRepAdaptor_Surface(face)
        if ad.GetType() == GeomAbs_Cylinder:
            r = round(ad.Cylinder().Radius(), 4)
            counts[r] = counts.get(r, 0) + 1
        exp.Next()
    if not counts:
        return None
    return max(counts, key=counts.get)


def build_inventory(solids: list, verbose: bool = True) -> list[Body]:
    """Measure every solid: volume, centroid, bounding box, class."""
    bodies: list[Body] = []
    t0 = time.time()

    for i, (solid, name, rgb) in enumerate(solids):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props)
        vol = props.Mass()
        com = props.CentreOfMass()

        bb = Bnd_Box()
        BRepBndLib.Add_s(solid, bb, False)
        if bb.IsVoid():
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()

        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        bodies.append(Body(
            index=i,
            name=name or f"unnamed_{i}",
            cls=classify(name or "", dx, dy, dz, vol),
            volume_mm3=vol,
            cx=com.X(), cy=com.Y(), cz=com.Z(),
            xmin=xmin, ymin=ymin, zmin=zmin,
            xmax=xmax, ymax=ymax, zmax=zmax,
            color=rgb,
            radius_mm=(cylinder_radius(solid)
                       if classify(name or "", dx, dy, dz, vol) == "cell" else None),
        ))

    if verbose:
        print(f"  measured {len(bodies)} bodies in {time.time() - t0:.1f}s")
    return bodies


def verify_units(bodies: list[Body]) -> dict:
    """Confirm the file really is in millimetres, using a cell as the ruler.

    An 18650 is 18 mm across and 65 mm long by definition.  If the cell bodies
    measure that, the file is mm; if they measured 0.018 / 0.065 it would be
    metres, and 0.71 / 2.56 would be inches.
    """
    cells = [b for b in bodies if b.cls == "cell"]
    if not cells:
        return {"ok": False, "reason": "no cells found to check against"}

    lengths = np.array([max(b.dx, b.dy, b.dz) for b in cells])
    diameters = np.array([sorted((b.dx, b.dy, b.dz))[0] for b in cells])

    med_len, med_dia = float(np.median(lengths)), float(np.median(diameters))
    ok = abs(med_len - CELL_LENGTH_MM) < 2.0 and abs(med_dia - CELL_DIAMETER_MM) < 2.0
    return {
        "ok": ok,
        "units": "mm" if ok else "UNCERTAIN",
        "n_cells": len(cells),
        "median_cell_length_mm": med_len,
        "median_cell_diameter_mm": med_dia,
        "expected_length_mm": CELL_LENGTH_MM,
        "expected_diameter_mm": CELL_DIAMETER_MM,
    }


def cell_axis(bodies: list[Body]) -> str:
    """Which global axis the cells' 65 mm dimension points along.

    This sets the flow orientation: air moves across the cells, i.e. in the
    plane perpendicular to this axis.
    """
    cells = [b for b in bodies if b.cls == "cell"]
    if not cells:
        return "?"
    votes = {"x": 0, "y": 0, "z": 0}
    for b in cells:
        dims = {"x": b.dx, "y": b.dy, "z": b.dz}
        votes[max(dims, key=dims.get)] += 1
    return max(votes, key=votes.get)


def write_inventory_csv(bodies: list[Body], path: pathlib.Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "index", "name", "class", "volume_mm3",
            "cx_mm", "cy_mm", "cz_mm",
            "dx_mm", "dy_mm", "dz_mm",
            "xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
        ])
        for b in bodies:
            w.writerow([
                b.index, b.name, b.cls, f"{b.volume_mm3:.3f}",
                f"{b.cx:.3f}", f"{b.cy:.3f}", f"{b.cz:.3f}",
                f"{b.dx:.3f}", f"{b.dy:.3f}", f"{b.dz:.3f}",
                f"{b.xmin:.3f}", f"{b.ymin:.3f}", f"{b.zmin:.3f}",
                f"{b.xmax:.3f}", f"{b.ymax:.3f}", f"{b.zmax:.3f}",
            ])


# =============================================================================
# Tessellation + cache
# =============================================================================

def tessellate(solid, tol: float = 0.5, angular_tol: float = 0.5):
    """Triangulate one solid; return (points Nx3, faces flat VTK array).

    Faces come back in VTK's flat layout: [3, i, j, k, 3, i, j, k, ...].
    """
    BRepMesh_IncrementalMesh(solid, tol, False, angular_tol, True)

    all_pts: list = []
    all_tris: list = []
    offset = 0

    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                all_pts.append((p.X(), p.Y(), p.Z()))

            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                a, b, c = t.Get()
                if reversed_face:
                    a, c = c, a
                all_tris.append((a - 1 + offset, b - 1 + offset, c - 1 + offset))
            offset += n_nodes
        exp.Next()

    if not all_pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    pts = np.asarray(all_pts, dtype=np.float64)
    tris = np.asarray(all_tris, dtype=np.int64)
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    return pts, faces


def _step_fingerprint(path: pathlib.Path) -> str:
    """Cheap but sound cache key: size + mtime + a hash of the file head."""
    st = path.stat()
    head = path.open("rb").read(65536)
    h = hashlib.sha256(head).hexdigest()[:16]
    return f"{st.st_size}_{int(st.st_mtime)}_{h}"


def load_or_build(path=None, tol: float = 0.5, force: bool = False,
                  verbose: bool = True) -> dict:
    """The one entry point: return {bodies, meshes, meta}, building the cache if needed.

    `path` accepts a source key ('legacy', 'box', 'full', 'pack') or an explicit
    path.  It defaults to `None` -> the original `BoxAssembly.step`, so the
    existing callers in inventory.py / render.py / simulate.py keep working
    without change.
    """
    step_path = resolve_source(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _step_fingerprint(step_path)
    # The stem is in the key so the three caches are human-identifiable at a
    # glance -- they run to 60-130 MB each and need pruning by hand sometimes.
    stem = step_path.stem.replace(" ", "_")
    cache_file = CACHE_DIR / f"geometry_{stem}_{fp}_tol{tol}.pkl.gz"

    if cache_file.exists() and not force:
        if verbose:
            print(f"cache hit: {cache_file.name}")
        with gzip.open(cache_file, "rb") as fh:
            return pickle.load(fh)

    if verbose:
        print(f"cache miss -- building from {step_path.name} "
              f"({step_path.stat().st_size / 1e6:.0f} MB)")

    shape_tool, color_tool, doc = read_step(step_path, verbose=verbose)
    solids = explode_assembly(shape_tool, color_tool, verbose=verbose)
    bodies = build_inventory(solids, verbose=verbose)

    if verbose:
        print(f"  tessellating at {tol} mm ...")
    t0 = time.time()
    meshes = []
    for i, (solid, _, _) in enumerate(solids):
        if i >= len(bodies):
            break
        meshes.append(tessellate(solid, tol))
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(solids)}  ({time.time() - t0:.0f}s)")
    if verbose:
        print(f"  tessellated in {time.time() - t0:.1f}s")

    data = {
        "bodies": bodies,
        "meshes": meshes,
        "meta": {
            "step": str(step_path),
            "fingerprint": fp,
            "tolerance_mm": tol,
            "units": verify_units(bodies),
            "cell_axis": cell_axis(bodies),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # gzip: the mesh arrays compress to roughly a fifth, and the extra
    # decompression costs far less than re-tessellating.
    with gzip.open(cache_file, "wb", compresslevel=4) as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"cached -> {cache_file.name} ({cache_file.stat().st_size / 1e6:.1f} MB)")
    return data


# =============================================================================
# Pack layout extraction -- what the flow model actually needs
# =============================================================================

def _cluster_1d(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """Group sorted scalars into clusters separated by more than `gap`."""
    order = np.argsort(values)
    v = values[order]
    splits = np.where(np.diff(v) > gap)[0]
    groups = np.split(order, splits + 1)
    return groups


def _can_diameter(cells: list[Body]) -> float:
    """Median measured can diameter [mm], falling back to the nominal 18650."""
    radii = [b.radius_mm for b in cells if b.radius_mm]
    if radii:
        return float(2.0 * np.median(radii))
    return CELL_DIAMETER_MM


def pack_layout(bodies: list[Body], verbose: bool = True) -> dict:
    """Recover the cell lattice: rows, columns, banks, pitches and stagger.

    This is the payload for the flow model.  The design review back-solved its
    geometry from a sizing script; these numbers come straight off the CAD.

    The lattice is resolved hierarchically rather than by clustering each axis
    globally.  A staggered bank has alternate rows shifted by half a pitch, so
    a global clustering of the cross-flow coordinate sees 2 x N interleaved
    positions at half the true pitch -- which is how a 12-wide bank on a
    20.5 mm pitch masquerades as 24 columns on a 10.25 mm pitch.  Grouping by
    row *first* and only then measuring within the row avoids that.
    """
    cells = [b for b in bodies if b.cls == "cell"]
    if not cells:
        return {"n_cells": 0}

    P = np.array([[b.cx, b.cy, b.cz] for b in cells])
    axis = cell_axis(bodies)
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    plane = [i for i in (0, 1, 2) if i != ai]

    # -- banks: stacked end to end along the cell axis ----------------------
    bank_groups = _cluster_1d(P[:, ai], gap=CELL_LENGTH_MM * 0.5)
    bank_centres = sorted(float(P[g, ai].mean()) for g in bank_groups)

    # -- decide which cross-flow axis indexes rows and which indexes columns.
    # Rows are the direction with more distinct levels and the coarser pitch;
    # for this pack that is the long axis of the box.
    counts = {}
    for pi in plane:
        groups = _cluster_1d(P[:, pi], gap=CELL_DIAMETER_MM * 0.45)
        counts[pi] = len(groups)
    row_axis = max(plane, key=lambda pi: counts[pi]) if counts[plane[0]] != counts[plane[1]] \
        else plane[1]
    # the axis with the *larger* span between distinct levels is the row axis;
    # break ties by span so the choice does not depend on clustering noise
    spans = {pi: float(P[:, pi].max() - P[:, pi].min()) for pi in plane}
    row_axis = max(plane, key=lambda pi: spans[pi])
    col_axis = [pi for pi in plane if pi != row_axis][0]

    # -- rows -----------------------------------------------------------------
    row_groups = _cluster_1d(P[:, row_axis], gap=CELL_DIAMETER_MM * 0.45)
    row_groups = sorted(row_groups, key=lambda g: P[g, row_axis].mean())
    row_centres = [float(P[g, row_axis].mean()) for g in row_groups]

    # -- columns, measured *within* each row ---------------------------------
    per_row = []
    for g in row_groups:
        xs = np.sort(np.unique(np.round(P[g, col_axis], 3)))
        per_row.append({
            "centre_mm": float(P[g, row_axis].mean()),
            "n_cells": int(len(g)),
            "n_cols": int(len(xs)),
            "first_mm": float(xs[0]),
            "pitch_mm": float(np.median(np.diff(xs))) if len(xs) > 1 else None,
        })

    col_pitches = [r["pitch_mm"] for r in per_row if r["pitch_mm"]]
    S_T = float(np.median(col_pitches)) if col_pitches else None
    S_L = float(np.median(np.diff(row_centres))) if len(row_centres) > 1 else None

    # -- stagger: do alternate rows start half a pitch apart? ----------------
    firsts = np.array([r["first_mm"] for r in per_row])
    offsets = np.abs(np.diff(firsts))
    stagger_offset = float(np.median(offsets)) if len(offsets) else 0.0
    staggered = bool(S_T and stagger_offset > 0.25 * S_T)

    layout = {
        "n_cells": len(cells),
        "cell_axis": axis,
        "row_axis": "xyz"[row_axis],
        "col_axis": "xyz"[col_axis],
        "n_rows": len(row_groups),
        "n_cols": int(np.median([r["n_cols"] for r in per_row])),
        "n_banks": len(bank_groups),
        "arrangement": "staggered" if staggered else "aligned",
        "stagger_offset_mm": stagger_offset,
        "transverse_pitch_S_T_mm": S_T,
        "longitudinal_pitch_S_L_mm": S_L,
        "bank_centres_mm": bank_centres,
        "bank_pitch_mm": (float(np.median(np.diff(bank_centres)))
                          if len(bank_centres) > 1 else None),
        "row_centres_mm": row_centres,
        "rows": per_row,
        # bounding-box diameter includes the wrap/tab features, so also report
        # the bare can diameter the 18650 designation implies
        "cell_bbox_diameter_mm": float(np.median(
            [sorted((b.dx, b.dy, b.dz))[0] for b in cells])),
        "cell_diameter_mm": _can_diameter(cells),
        "cell_length_mm": float(np.median([max(b.dx, b.dy, b.dz) for b in cells])),
        "cell_span_mm": {
            "row": [float(P[:, row_axis].min()), float(P[:, row_axis].max())],
            "col": [float(P[:, col_axis].min()), float(P[:, col_axis].max())],
            "bank": [float(P[:, ai].min()), float(P[:, ai].max())],
        },
        "positions_mm": P.tolist(),
    }

    if S_T and S_L:
        # the can, not the bounding box -- see `cylinder_radius`
        D = layout["cell_diameter_mm"]
        S_D = math.hypot(S_L, S_T / 2.0)
        layout["diagonal_pitch_S_D_mm"] = S_D
        layout["S_T_over_D"] = S_T / D
        layout["S_L_over_D"] = S_L / D
        layout["transverse_gap_mm"] = S_T - D
        layout["diagonal_gap_mm"] = S_D - D
        layout["min_gap_mm"] = min(S_T - D, 2.0 * (S_D - D))

    return layout


def case_envelope(bodies: list[Body]) -> dict:
    """Inner extent of the case, from the plate bodies."""
    plates = [b for b in bodies if b.cls == "case_plate"]
    if not plates:
        return {}
    return {
        "n_plates": len(plates),
        "xmin": min(b.xmin for b in plates), "xmax": max(b.xmax for b in plates),
        "ymin": min(b.ymin for b in plates), "ymax": max(b.ymax for b in plates),
        "zmin": min(b.zmin for b in plates), "zmax": max(b.zmax for b in plates),
        "names": sorted({b.name for b in plates}),
    }


def _main() -> None:
    data = load_or_build()
    bodies = data["bodies"]
    print()
    print(f"bodies: {len(bodies)}")
    from collections import Counter
    for cls, n in Counter(b.cls for b in bodies).most_common():
        print(f"  {cls:12s} {n}")
    print()
    print("units:", json.dumps(data["meta"]["units"], indent=2))
    print()
    print("layout:", json.dumps({k: v for k, v in pack_layout(bodies).items()
                                 if k != "positions_mm"}, indent=2))


if __name__ == "__main__":
    # Run as a module so pickled Body objects resolve to `geometry.Body`
    # rather than `__main__.Body`, which would make the cache unreadable
    # from any other import site.
    if __package__ is None and "geometry" not in __import__("sys").modules:
        import sys, pathlib as _pl
        sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
        import geometry as _g
        _g._main()
    else:
        _main()
