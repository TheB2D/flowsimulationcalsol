# Battery Pack Airflow & Thermal Modelling — Project Context

## Goal

Model cooling airflow and cell temperatures inside a CalSol battery pack enclosure
(reference: `Excalibur Battery Design Review.pdf`). Two deliverables right now:

1. **Interactive 3D render** of the geometry in Python (case + battery pack).
2. **Reduced-order airflow + thermal model** (fan curve → operating point → air/cell temperatures).

Full CFD (OpenFOAM / FEniCSx) comes later, see the last section.

The user prefers math written in proper notation (LaTeX) in explanations.

---

## Working directory

`/Users/christopherkhaing/Desktop/fan curve stuff/`

```
fan curve stuff/
├── BoxAssembly.step                 # PRIMARY INPUT — exported geometry (case + battery pack)
├── 8-0005 Box_assembly/             # original SolidWorks files — READ ONLY, do not modify
│   ├── *.SLDPRT, *.SLDASM           # (includes renamed copies BoxAssembly.SLDASM, BatteryPack.SLDASM)
│   └── Excalibur Battery Design Review.pdf   # design reference (pack layout, fans, airflow path)
├── CalSol Battery Fall 2024 Onboarding Presentation.pdf
├── flow_model/                      # EXISTING prior work — inspect before writing new code
├── design-system/                   # unknown contents — check before use
└── *.zip                            # Onshape upload zips — ignore
```

---

## Geometry provenance

SolidWorks (native `.SLDASM`) → Onshape (native import) → **STEP AP242** export.
SolidWorks files cannot be read with open-source tools; **the STEP file is the source of truth.**

### What's in `BoxAssembly.step`

- **Case (7 plates):** `8-0001 Box_Bottom_plate` ×1, `8-0002 Box_Back_plate` ×2,
  `8-0003 Box_Side_plate` ×2, `8-0004 Box_Top_piece` ×2. The top has a large rectangular opening.
- **BatteryPack subassembly:** multiple `Battery Module` instances (at least 7), built from
  LG INR18650 MJ1 cells plus busbars, divider boards, tie rods, cell mounting tape, tab sheets/solder,
  module spacers, etc. (exact membership unverified; inspect the STEP).
- **Rideon SSD-500C mockup:** small part, left in.

### Deliberately excluded (suppressed before export)

- 4× Gigavac GX11 contactors, CHSF-70 fuse
- Fan/duct unit `8-00008-P-XX` (Delta GFB0812ES-E fans, fan grille, rear fan duct, exhaust duct)

### Caveats — read before trusting the geometry

- **Pack placement is approximate.** The original assembly relating box and pack was not available.
  The pack was inserted and rotated/positioned by hand in Onshape (no mates), then fixed.
  Sanity-check that the pack sits inside the case with no interference, and compare against the design review PDF.
- **Fans are not in the model.** Inlet/outlet locations must be identified from the case openings
  and the design review PDF. Ask the user to confirm the airflow path before building the flow model.
- **Units:** verify. Onshape STEP is likely mm. Check with an 18650 cell's bounding box
  (≈ 18 mm diameter × 65 mm length).
- **Size:** hundreds of solids (every cell is a body). Tessellation is slow, so cache it.

---

## Task 1 — 3D render

Environment: macOS (Apple Silicon likely). Install:

```bash
pip install cadquery pyvista numpy
# fallback if cadquery pip install fails:
# conda install -c conda-forge cadquery pyvista
```

Requirements:

- Load the STEP, iterate over solids, tessellate (~0.5 mm tolerance), display in an interactive PyVista window.
- **Classify bodies** by bounding box / volume: `cell` (cylinder ≈ 18 × 65 mm), `case_plate`, `other`.
  Color by class; make the case translucent so the cells are visible.
- **Cache** the tessellated meshes (e.g. `.vtm` or `.npz`) so re-rendering is fast.
- Write a **body inventory CSV**: index, class, volume, centroid, bounding box.
  This feeds the flow model (cell count, cell positions, module grouping, gap spacing).
- Optional: `plotter.export_html()` for a shareable browser view.
- The render must later be able to **color cells by computed temperature**.

Starter:

```python
import cadquery as cq, numpy as np, pyvista as pv

shape = cq.importers.importStep("BoxAssembly.step")
p = pv.Plotter()
for s in shape.Solids():
    verts, tris = s.tessellate(0.5)
    pts = np.array([[v.x, v.y, v.z] for v in verts])
    faces = np.hstack([[3, *t] for t in tris])
    p.add_mesh(pv.PolyData(pts, faces), color=np.random.rand(3))
p.show()
```

---

## Task 2 — Reduced-order airflow + thermal model

### Inputs (do NOT invent values; ask the user or leave as clearly labelled parameters with units + source)

- Fan curve $\Delta p_{\text{fan}}(Q)$ for the Delta GFB0812ES-E, plus the number of fans and their arrangement.
  The user may already have this; check `flow_model/` and ask.
- Pack electrical configuration (series × parallel), design current $I$ (or current profile).
- Cell internal resistance $R_{\text{int}}$ (LG MJ1 datasheet).
- Ambient temperature $T_\infty$ and maximum allowed cell temperature.
- Flow path geometry: inlet/outlet location and area, and cell pitch/gaps (from the Task 1 inventory).

### Model

1. **Fan operating point:** solve
   $$\Delta p_{\text{fan}}(Q) = \Delta p_{\text{sys}}(Q), \qquad \Delta p_{\text{sys}}(Q) \approx K Q^2$$
   Estimate $K$ from tube-bank pressure-drop correlations for cross-flow over the cell array, plus
   entry/exit losses. Keep $K$ easy to recalibrate against CFD later.
2. **Heat generation per cell:** $\dot{Q}_{\text{cell}} = I_{\text{cell}}^2 R_{\text{int}}$,
   with $I_{\text{cell}} = I / N_p$.
3. **Air temperature rise** marching along the flow path (row by row):
   $$\dot{m}\, c_p\, \Delta T_{\text{air}} = \sum \dot{Q}_{\text{cell}}$$
4. **Cell temperature:**
   $$T_{\text{cell}} = T_{\text{air,local}} + \frac{\dot{Q}_{\text{cell}}}{h A_{\text{cell}}}$$
   Take $h$ from a Nusselt correlation for cross-flow over cylinder banks (e.g. Žukauskas), using $Re$
   based on the maximum velocity in the cell gaps and $D \approx 18$ mm.
5. **Outputs:**
   - Plot of fan curve vs system curve with the operating point marked
   - Per-cell temperature table
   - The 3D render with cells colored by temperature
6. **Sanity checks:** global energy balance, and $Re$ within each correlation's validity range.

### Suggested code structure

- `geometry.py`: STEP loading, classification, inventory, cache
- `render.py`: PyVista views (plain + temperature-colored)
- `flow.py`: operating point, air march, cell temperatures
- `params.py` (or `.yaml`): every physical parameter with units and source noted

---

## Later — full CFD / conduction

- **OpenFOAM** `chtMultiRegionFoam` (conjugate heat transfer), run via Docker on macOS and driven by Python.
  Meshing the thin gaps between cells is the hard part.
- **Gmsh + FEniCSx** for detailed conduction in the pack, using $h$ from the reduced-order model as the boundary condition:
  $$-k\,\frac{\partial T}{\partial n} = h\,(T - T_\infty)$$
- **Simplification option:** treat each module as a homogenized block with anisotropic conductivity
  $k_\parallel \neq k_\perp$ (along vs. across the cell axis) and a volumetric heat source.
- **SimScale** (cloud CFD with an Onshape integration) is an alternative, but free-plan projects are public,
  so check with the team before uploading CalSol CAD.

---

## Conventions

- Never modify the original SolidWorks files or the PDFs.
- Inspect `flow_model/` first and build on it instead of duplicating.
- Keep all physical parameters in one place, with units and sources.
- Ask before assuming anything about the airflow path, fan count, or electrical configuration.
