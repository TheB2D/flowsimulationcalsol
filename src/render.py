"""
Interactive 3D views of the battery box.

Two modes:

    python src/render.py                 # classified view, case translucent
    python src/render.py --temperature   # cells coloured by temperature
    python src/render.py --html          # write a shareable out/render.html

The temperature mode takes an array of 408 cell temperatures indexed the same
way as `out/cell_positions.csv` (row-major: row, then col, then bank).  Until
the flow model in Task 2 supplies real ones, `--temperature` with no data file
shows a synthetic front-to-back gradient so the colour path is exercised and
the plumbing is proven before the physics arrives.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as G
from geometry import Body


# Colours by class.  Opacity is set so the cells stay visible: the divider
# boards and module spacers wrap around the cells and would hide them entirely
# at any substantial opacity, so they are dropped right back.  The cells are
# the only class drawn solid.
CLASS_STYLE = {
    "cell":       {"color": "#3B82F6", "opacity": 1.00},
    "case_plate": {"color": "#94A3B8", "opacity": 0.10},
    "busbar":     {"color": "#F59E0B", "opacity": 0.55},
    "structure":  {"color": "#64748B", "opacity": 0.10},
    "fastener":   {"color": "#475569", "opacity": 0.10},
    "other":      {"color": "#A78BFA", "opacity": 0.15},
}

# Draw order matters for transparency: opaque cells first, then the shells.
DRAW_ORDER = ["cell", "busbar", "other", "structure", "fastener", "case_plate"]


def _n_faces(mesh: pv.PolyData) -> int:
    """Triangle count, across PyVista versions."""
    return int(getattr(mesh, "n_cells", 0))


def build_blocks(data: dict, decimate: dict[str, float] | None = None
                 ) -> dict[str, pv.PolyData]:
    """Merge the per-solid meshes into one PolyData per class.

    2804 separate actors would crawl; six merged blocks render instantly and
    still let each class keep its own colour and opacity.

    `decimate` optionally thins a class to a fraction of its triangles.  The
    structure class alone is ~3.5 M triangles of divider boards and spacers
    that are drawn at 10% opacity anyway, so for a shareable export it can be
    cut hard without any visible change.  Cells are never decimated -- they
    carry the result.
    """
    bodies: list[Body] = data["bodies"]
    meshes = data["meshes"]

    buckets: dict[str, list[pv.PolyData]] = {}
    for b, (pts, faces) in zip(bodies, meshes):
        if len(pts) == 0:
            continue
        buckets.setdefault(b.cls, []).append(pv.PolyData(pts, faces))

    blocks = {}
    for cls, parts in buckets.items():
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        frac = (decimate or {}).get(cls)
        if frac and cls != "cell" and _n_faces(merged) > 20_000:
            try:
                merged = merged.triangulate().decimate(1.0 - frac)
            except Exception:
                pass    # decimation is cosmetic; never let it break the render
        blocks[cls] = merged
    return blocks


def cell_meshes(data: dict) -> tuple[list[pv.PolyData], list[Body]]:
    """Just the cells, kept separate so each can take its own scalar."""
    out_m, out_b = [], []
    for b, (pts, faces) in zip(data["bodies"], data["meshes"]):
        if b.cls == "cell" and len(pts):
            out_m.append(pv.PolyData(pts, faces))
            out_b.append(b)
    return out_m, out_b


def _ordered_cells(data: dict) -> list[Body]:
    """Cells in the same (row, col, bank) order as out/cell_positions.csv."""
    import inventory as I
    layout = G.pack_layout(data["bodies"])
    idx = I.index_cells(data["bodies"], layout)
    by_index = {b.index: b for b in data["bodies"]}
    return [by_index[c["body_index"]] for c in idx]


def scene(data: dict, temperatures: np.ndarray | None = None,
          show_case: bool = True, off_screen: bool = False,
          hide: tuple[str, ...] = (),
          decimate: dict[str, float] | None = None) -> pv.Plotter:
    """Assemble the plotter.

    `temperatures`: optional length-408 array in cell_positions.csv order.
    `hide`: class names to leave out entirely (e.g. to see the bank alone).
    """
    p = pv.Plotter(off_screen=off_screen, window_size=(1600, 1000))
    p.set_background("#0B1220")

    blocks = build_blocks(data, decimate=decimate)

    if temperatures is None:
        for cls in [c for c in DRAW_ORDER if c in blocks] + \
                   [c for c in blocks if c not in DRAW_ORDER]:
            mesh = blocks[cls]
            if cls == "case_plate" and not show_case:
                continue
            if cls in hide:
                continue
            st = CLASS_STYLE.get(cls, CLASS_STYLE["other"])
            p.add_mesh(mesh, color=st["color"], opacity=st["opacity"],
                       smooth_shading=True, label=f"{cls} ({_n_faces(mesh):,} tris)")
        p.add_legend(bcolor="#111827", face="rectangle", size=(0.18, 0.22))
        title = "CalSol Excalibur battery box - classified geometry"
    else:
        # everything except the cells, dimmed back so the colour map leads
        for cls, mesh in blocks.items():
            if cls == "cell":
                continue
            if cls == "case_plate" and not show_case:
                continue
            if cls in hide:
                continue
            st = CLASS_STYLE.get(cls, CLASS_STYLE["other"])
            p.add_mesh(mesh, color=st["color"],
                       opacity=min(st["opacity"], 0.18), smooth_shading=True)

        ordered = _ordered_cells(data)
        if len(temperatures) != len(ordered):
            raise ValueError(
                f"got {len(temperatures)} temperatures for {len(ordered)} cells")

        meshes, bods = cell_meshes(data)
        by_index = {b.index: m for b, m in zip(bods, meshes)}

        coloured = []
        for b, t in zip(ordered, temperatures):
            m = by_index.get(b.index)
            if m is None:
                continue
            m = m.copy()
            m["T"] = np.full(m.n_points, float(t))
            coloured.append(m)
        merged = coloured[0].merge(coloured[1:]) if len(coloured) > 1 else coloured[0]

        p.add_mesh(merged, scalars="T", cmap="inferno", smooth_shading=True,
                   scalar_bar_args={
                       "title": "Cell temperature  [C]",
                       "color": "white", "n_labels": 6,
                       "title_font_size": 16, "label_font_size": 13,
                   })
        title = (f"Cell temperature  -  {temperatures.min():.1f} to "
                 f"{temperatures.max():.1f} C")

    p.add_text(title, position="upper_left", font_size=11, color="white")
    p.add_axes(line_width=2, color="white")
    p.enable_anti_aliasing()
    p.camera_position = "iso"
    return p


def load_temperatures(path: pathlib.Path | None, n: int) -> np.ndarray:
    """Read cell temperatures, or synthesise a placeholder gradient.

    The placeholder is explicitly *not* physics -- it exists so the colour
    pipeline can be verified before Task 2 exists.  Real values will come from
    the flow model's row-by-row march.
    """
    if path and path.exists():
        if path.suffix == ".json":
            vals = json.loads(path.read_text())
            if isinstance(vals, dict):
                vals = vals.get("cell_temperatures_c", vals.get("temperatures"))
            return np.asarray(vals, dtype=float)
        return np.loadtxt(path, delimiter=",")

    print("  no temperature file given -- showing a PLACEHOLDER gradient, "
          "not a computed field")
    rows = 17
    per_row = n // rows
    ramp = np.linspace(40.0, 46.0, rows)
    return np.repeat(ramp, per_row)[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature", action="store_true",
                    help="colour cells by temperature instead of by class")
    ap.add_argument("--temps", type=pathlib.Path, default=None,
                    help="CSV/JSON of 408 cell temperatures in cell_positions.csv order")
    ap.add_argument("--no-case", action="store_true", help="hide the enclosure")
    ap.add_argument("--cells-only", action="store_true",
                    help="show only the cell bank (and the case outline)")
    ap.add_argument("--html", action="store_true",
                    help="write out/render.html instead of opening a window")
    ap.add_argument("--png", action="store_true", help="write out/render.png")
    ap.add_argument("--tol", type=float, default=0.5, help="tessellation tolerance, mm")
    ap.add_argument("--light", action="store_true",
                    help="decimate the non-cell classes -- much smaller HTML export")
    args = ap.parse_args()

    data = G.load_or_build(tol=args.tol)
    n_cells = sum(1 for b in data["bodies"] if b.cls == "cell")

    temps = None
    if args.temperature:
        temps = load_temperatures(args.temps, n_cells)

    off = args.html or args.png
    hide = ("structure", "fastener", "busbar", "other") if args.cells_only else ()
    # a full-detail HTML export runs to ~100 MB, which is not shareable, so the
    # browser view thins the classes that are drawn nearly transparent anyway
    dec = None
    if args.light or (args.html and not args.cells_only):
        dec = {"structure": 0.06, "busbar": 0.12, "fastener": 0.06, "other": 0.12}
    p = scene(data, temperatures=temps, show_case=not args.no_case,
              off_screen=off, hide=hide, decimate=dec)

    G.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.html:
        stem = "render_temperature" if args.temperature else "render"
        out = G.OUT_DIR / f"{stem}.html"
        try:
            p.export_html(str(out))
        except ImportError:
            # export_html goes through trame, which needs a server component
            # that is not always present.  The VTK.js scene export is
            # self-contained and opens in the standard vtk.js viewer, so fall
            # back to it rather than failing the run.
            out = G.OUT_DIR / f"{stem}.vtksz"
            p.export_vtksz(str(out))
            print("  (trame unavailable -- exported a vtk.js scene instead; "
                  "open it at https://kitware.github.io/vtk-js/examples/SceneExplorer.html)")
        print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    elif args.png:
        out = G.OUT_DIR / ("render_temperature.png" if args.temperature else "render.png")
        p.screenshot(str(out))
        print(f"wrote {out}")
    else:
        p.show()


if __name__ == "__main__":
    main()
