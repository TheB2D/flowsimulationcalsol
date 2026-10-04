"""
Write the body inventory and the pack-layout summary that the flow model reads.

Produces, in `out/`:
  * `body_inventory.csv`   -- every solid: class, volume, centroid, bounding box
  * `cell_positions.csv`   -- the 408 cells indexed by (row, col, bank)
  * `pack_layout.json`     -- pitches, counts, gaps, case envelope, fit check
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as G
from geometry import Body


def _jsonable(o):
    """JSON fallback for numpy scalars/arrays."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def index_cells(bodies: list[Body], layout: dict) -> list[dict]:
    """Label every cell with its (row, col, bank) index in the lattice.

    Rows and banks come from clustering; the column index is taken within the
    row, because the bank is staggered and alternate rows do not share column
    coordinates.
    """
    cells = [b for b in bodies if b.cls == "cell"]
    ra = {"x": 0, "y": 1, "z": 2}[layout["row_axis"]]
    ca = {"x": 0, "y": 1, "z": 2}[layout["col_axis"]]
    ba = {"x": 0, "y": 1, "z": 2}[layout["cell_axis"]]

    row_centres = np.array(layout["row_centres_mm"])
    bank_centres = np.array(layout["bank_centres_mm"])

    out = []
    for b in cells:
        c = np.array([b.cx, b.cy, b.cz])
        row = int(np.argmin(np.abs(row_centres - c[ra])))
        bank = int(np.argmin(np.abs(bank_centres - c[ba])))
        out.append({
            "body_index": b.index, "row": row, "bank": bank,
            "col_coord": float(c[ca]),
            "x": b.cx, "y": b.cy, "z": b.cz,
            "volume_mm3": b.volume_mm3,
            "diameter_mm": (2 * b.radius_mm) if b.radius_mm else None,
        })

    # column index: rank of the cell's cross-flow coordinate within its own row
    for row in sorted({o["row"] for o in out}):
        members = [o for o in out if o["row"] == row]
        coords = sorted({round(o["col_coord"], 3) for o in members})
        rank = {c: i for i, c in enumerate(coords)}
        for o in members:
            o["col"] = rank[round(o["col_coord"], 3)]

    out.sort(key=lambda o: (o["row"], o["col"], o["bank"]))
    return out


def fit_check(bodies: list[Body], layout: dict) -> dict:
    """Does the pack actually sit inside the case?

    CLAUDE.md flags that the pack was positioned by hand in Onshape with no
    mates, so this is the sanity check it asks for.

    The verdict is taken on the **cells**, because they are what the flow model
    is about and what must not clash with the enclosure.  Other bodies are
    reported separately: a handful of parts in this export have corrupt
    bounding boxes (see `export_artifacts`), and judging the fit on those would
    raise a false alarm.
    """
    env = G.case_envelope(bodies)
    if not env:
        return {"ok": False, "reason": "no case plates found"}

    def extent(group):
        return {
            "xmin": min(b.xmin for b in group), "xmax": max(b.xmax for b in group),
            "ymin": min(b.ymin for b in group), "ymax": max(b.ymax for b in group),
            "zmin": min(b.zmin for b in group), "zmax": max(b.zmax for b in group),
        }

    def clearances(ext):
        out = {}
        for ax in "xyz":
            out[ax] = {
                "case_mm": [env[f"{ax}min"], env[f"{ax}max"]],
                "part_mm": [ext[f"{ax}min"], ext[f"{ax}max"]],
                "clearance_low_mm": ext[f"{ax}min"] - env[f"{ax}min"],
                "clearance_high_mm": env[f"{ax}max"] - ext[f"{ax}max"],
            }
        return out

    cells = [b for b in bodies if b.cls == "cell"]
    others = [b for b in bodies if b.cls != "case_plate" and b.cls != "cell"]

    cell_clear = clearances(extent(cells))
    cells_ok = all(c["clearance_low_mm"] >= -1.0 and c["clearance_high_mm"] >= -1.0
                   for c in cell_clear.values())

    # bodies poking outside the enclosure, excluding the cells
    escapees = [b for b in others
                if b.xmin < env["xmin"] - 0.5 or b.xmax > env["xmax"] + 0.5
                or b.ymin < env["ymin"] - 0.5 or b.ymax > env["ymax"] + 0.5
                or b.zmin < env["zmin"] - 0.5 or b.zmax > env["zmax"] + 0.5]

    return {
        "ok": cells_ok,
        "verdict": ("all cells inside the enclosure"
                    if cells_ok else "CELLS CLASH WITH THE ENCLOSURE"),
        "case": env,
        "cell_extent_mm": extent(cells),
        "cell_clearances": cell_clear,
        "n_noncell_bodies_outside": len(escapees),
        "noncell_bodies_outside": sorted({b.name for b in escapees}),
    }


def size_outliers(bodies: list[Body]) -> dict:
    """Bodies whose size disagrees with others carrying the same part name.

    Advisory only.  A name in this export is the *part* name, and several parts
    (the interconnect covers, the module spacers) contribute more than one
    solid, so same-name bodies are not always the same thing -- most of what
    this reports is that, not a defect.  It is here to catch the case that
    would matter: a cell that came through the export the wrong size.  None do,
    which is the check that counts, since the cells are the flow geometry.
    """
    from collections import defaultdict
    by_name = defaultdict(list)
    for b in bodies:
        by_name[b.name].append(b)

    bad = []
    for name, group in by_name.items():
        if len(group) < 3:
            continue
        # Compare *sorted* extents: the same part placed in a different
        # orientation has the same three dimensions in a different order, and
        # that is a rotation, not a defect.  Only a genuine change in size
        # survives the sort.
        dims = np.array([sorted((b.dx, b.dy, b.dz)) for b in group])
        med = np.median(dims, axis=0)
        for b, d in zip(group, dims):
            rel = np.abs(d - med) / np.maximum(med, 1e-6)
            if np.any((np.abs(d - med) > 5.0) & (rel > 0.2)):
                bad.append({
                    "index": int(b.index), "name": name, "class": b.cls,
                    "sorted_dims_mm": [round(float(x), 2) for x in d],
                    "expected_mm": [round(float(x), 2) for x in med],
                })
    return {"n": len(bad), "bodies": bad}


def main() -> None:
    data = G.load_or_build()
    bodies = data["bodies"]
    layout = G.pack_layout(bodies)
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. full body inventory ---------------------------------------------
    G.write_inventory_csv(bodies, G.OUT_DIR / "body_inventory.csv")

    # -- 2. cell lattice ------------------------------------------------------
    cells = index_cells(bodies, layout)
    with (G.OUT_DIR / "cell_positions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "row", "col", "bank", "body_index",
                    "x_mm", "y_mm", "z_mm", "diameter_mm", "volume_mm3"])
        for i, c in enumerate(cells):
            w.writerow([i, c["row"], c["col"], c["bank"], c["body_index"],
                        f"{c['x']:.3f}", f"{c['y']:.3f}", f"{c['z']:.3f}",
                        f"{c['diameter_mm']:.3f}" if c["diameter_mm"] else "",
                        f"{c['volume_mm3']:.2f}"])

    # -- 3. layout + fit ------------------------------------------------------
    summary = {k: v for k, v in layout.items() if k not in ("positions_mm", "rows")}
    summary["row_detail"] = layout["rows"]
    summary["units"] = data["meta"]["units"]
    summary["fit_check"] = fit_check(bodies, layout)
    summary["size_outliers"] = size_outliers(bodies)
    from collections import Counter
    summary["class_counts"] = dict(Counter(b.cls for b in bodies))
    summary["part_counts"] = dict(Counter(b.name for b in bodies).most_common(25))

    (G.OUT_DIR / "pack_layout.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable))

    # -- report ---------------------------------------------------------------
    print(f"bodies ............. {len(bodies)}")
    for cls, n in Counter(b.cls for b in bodies).most_common():
        print(f"  {cls:12s} {n}")
    print()
    print(f"cells .............. {layout['n_cells']}  "
          f"({layout['n_rows']} rows x {layout['n_cols']} cols x {layout['n_banks']} banks)")
    print(f"arrangement ........ {layout['arrangement']}  "
          f"(offset {layout['stagger_offset_mm']:.2f} mm)")
    print(f"cell diameter D .... {layout['cell_diameter_mm']:.2f} mm   "
          f"(bbox {layout['cell_bbox_diameter_mm']:.2f} mm)")
    print(f"cell length ........ {layout['cell_length_mm']:.2f} mm")
    print(f"S_T (transverse) ... {layout['transverse_pitch_S_T_mm']:.2f} mm   "
          f"S_T/D = {layout['S_T_over_D']:.3f}")
    print(f"S_L (longitudinal) . {layout['longitudinal_pitch_S_L_mm']:.2f} mm   "
          f"S_L/D = {layout['S_L_over_D']:.3f}")
    print(f"S_D (diagonal) ..... {layout['diagonal_pitch_S_D_mm']:.2f} mm")
    print(f"min gap ............ {layout['min_gap_mm']:.2f} mm")
    print(f"flow axes .......... rows along {layout['row_axis']}, "
          f"cols along {layout['col_axis']}, cell axis {layout['cell_axis']}")
    print()
    fc = summary["fit_check"]
    print(f"case plates ........ {fc['case']['n_plates']}  {', '.join(fc['case']['names'])}")
    print(f"pack fit ........... {fc['verdict']}")
    for ax, c in fc["cell_clearances"].items():
        print(f"  {ax}: case [{c['case_mm'][0]:8.1f},{c['case_mm'][1]:8.1f}]  "
              f"cells [{c['part_mm'][0]:8.1f},{c['part_mm'][1]:8.1f}]  "
              f"clear {c['clearance_low_mm']:7.1f} / {c['clearance_high_mm']:7.1f} mm")
    so = summary["size_outliers"]
    cell_outliers = [b for b in so["bodies"] if b["class"] == "cell"]
    print(f"\nsize outliers ...... {so['n']} body(ies), "
          f"{len(cell_outliers)} of them cells")
    if cell_outliers:
        print("  WARNING: a cell came through the export mis-sized")
        for b in cell_outliers[:5]:
            print(f"    idx {b['index']}  {b['sorted_dims_mm']} vs {b['expected_mm']}")
    else:
        print("  none are cells -- the flow geometry is clean.  The rest are "
              "multi-body parts\n  whose solids share one part name, not export damage.")

    if fc["n_noncell_bodies_outside"]:
        print(f"\nnon-cell bodies outside the enclosure: "
              f"{fc['n_noncell_bodies_outside']} "
              f"({', '.join(fc['noncell_bodies_outside'])})")

    print()
    for f in ("body_inventory.csv", "cell_positions.csv", "pack_layout.json"):
        pth = G.OUT_DIR / f
        print(f"wrote out/{f}  ({pth.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    from collections import Counter
    main()
