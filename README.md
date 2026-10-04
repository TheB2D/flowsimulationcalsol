# CalSol Excalibur — pack geometry, airflow and thermal model

## Task 1 — 3D render and geometry extraction

Reads `BoxAssembly.step`, classifies every solid, writes a body inventory, and
renders the pack interactively — including colouring cells by temperature, which
is what Task 2 will drive.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install cadquery pyvista numpy scipy trame-pyvista nest_asyncio2

.venv/bin/python src/geometry.py    # read + classify + tessellate (first run ~33 s, then ~0.3 s)
.venv/bin/python src/inventory.py   # write out/*.csv and out/pack_layout.json
.venv/bin/python src/render.py      # interactive window
```

```bash
.venv/bin/python src/simulate.py   # Task 2: flow + thermal, writes results and figures
.venv/bin/python src/render.py --temperature --temps out/cell_temperatures.json
```

Render options: `--temperature` (colour cells by a temperature field), `--temps FILE`
(408 values, `cell_positions.csv` order), `--cells-only`, `--no-case`, `--html`,
`--png`, `--light`, `--tol MM`.

## What the STEP turned out to be

`importStep(...).Solids()` — the snippet in `CLAUDE.md` — does not work here, for
two reasons. The current CadQuery returns a `Workplane`, so the call is
`.val().Solids()`; and more importantly it throws away the assembly tree. This
file is read instead through the **XCAF** document reader, which keeps the part
name and colour on every solid. Classification then keys off the CAD part names
rather than guessing from bounding boxes, and the names are what tie each body
back to the BOM.

The file is a **hybrid** AP242 export: 70 unique B-rep part bodies, instanced by
3162 assembly occurrences into **2804 placed solids**. Tessellating the unique
parts and reusing them is why the cold build is 33 s rather than tens of minutes.

## Inventory

| class | count | what it is |
|---|---:|---|
| `busbar` | 949 | busbars, interconnects, terminals, tabs |
| `structure` | 878 | divider boards, module spacers, tie rods, tape, covers |
| `cell` | 408 | LG INR18650 MJ1 |
| `other` | 389 | unclassified small parts |
| `fastener` | 173 | screws, washers |
| `case_plate` | 7 | the enclosure — matches the 1 + 2 + 2 + 2 in `CLAUDE.md` |

## Pack geometry, measured from CAD

| quantity | value |
|---|---|
| cells | **408** = 17 rows × 12 across × 2 banks |
| arrangement | **staggered**, alternate rows offset by exactly $S_T/2 = 10.25$ mm |
| cell diameter $D$ | **18.40 mm** |
| cell length | **65.00 mm** |
| transverse pitch $S_T$ | **20.50 mm** &nbsp;($S_T/D = 1.114$) |
| longitudinal pitch $S_L$ | **25.20 mm** &nbsp;($S_L/D = 1.370$) |
| diagonal pitch $S_D$ | **27.20 mm** |
| minimum gap between cells | **2.10 mm** |
| axes | rows along $z$, columns along $x$, cell axis along $y$ |

### Units

Confirmed as **millimetres**, checked against the thing `CLAUDE.md` asks for: the
cells measure 65.00 mm long and 18.40 mm across, which is an 18650 by definition.

The 18.40 mm is measured off the **cylindrical face**, not the bounding box. The
bounding box reads 19.92 mm because it includes the terminal button and the tab.
The flow model needs the can, since that is the cylinder the air actually sees —
and the difference is not cosmetic: it moves the inter-cell gap from 0.58 mm to
2.10 mm, a factor of 3.6 in the quantity that sets the pressure drop.

### Pack placement

`CLAUDE.md` warns that the pack was positioned by hand in Onshape with no mates.
Checked: **all 408 cells sit inside the enclosure**, with 217/178 mm clearance in
$x$, 37/11 mm in $y$ and 49/28 mm in $z$. No interference.

51 non-cell bodies (module spacers, divider boards) do report as outside the box,
but that is a naming artifact of this export — several multi-body parts give all
their solids one part name — not a placement problem. `size_outliers()` in
`src/inventory.py` reports these, and confirms **none of them are cells**, which
is the check that matters for the flow model.

## How this compares with the existing `flow_model/`

The prior work in `~/Desktop/fan curve stuff/flow_model/` could not open the CAD,
so it reconstructed the geometry by back-solving the design review's sizing
script. Its README flags that reconstruction as its largest open assumption.
Measuring the STEP settles it, and the reconstruction was mostly right:

| | reconstructed | measured | |
|---|---|---|---|
| cells | 408 | **408** | ✓ |
| layout | 17 × 12 × 2 | **17 × 12 × 2** | ✓ |
| arrangement | staggered | **staggered** | ✓ |
| $D$ | 18 mm | **18.4 mm** | ✓ |
| $S_T$ | 20.60 mm | **20.50 mm** | ✓ within 0.5% |
| $S_L$ | 20.60 mm | **25.20 mm** | ✗ **22% low** |

**The one substantive correction is $S_L$.** `pack_model.py` sets
`longitudinal_pitch = transverse_pitch`, i.e. assumes an equilateral staggered
bank, and notes `chi = 1.0` on the grounds that $S_T = S_L$. The CAD says the bank
is not equilateral: $S_L = 25.20$ mm against $S_T = 20.50$ mm, so $S_L/S_T = 1.23$.

That matters for Task 2 in two places:

- the **diagonal pitch** becomes $S_D = \sqrt{S_L^2 + (S_T/2)^2} = 27.20$ mm, so the
  diagonal gap ($2(S_D - D) = 17.6$ mm) is far wider than the transverse gap
  (2.10 mm). The transverse plane is unambiguously the minimum-area plane, which
  simplifies $V_{\max}$ — `pack_model.py` already takes the `min` of the two, so it
  gets the right answer, but for the right reason only once $S_L$ is corrected.
- the Žukauskas friction correction $\chi$ is **not** 1.0 for a non-equilateral bank,
  and $\Delta p = N_L \chi f (\rho V_{\max}^2 / 2)$ depends on it directly.

Both should be revisited when the flow model is rebuilt on this geometry.

## Outputs

| file | contents |
|---|---|
| `out/body_inventory.csv` | 2804 rows — index, name, class, volume, centroid, bounding box |
| `out/cell_positions.csv` | 408 rows — `cell_id`, `(row, col, bank)`, position, diameter |
| `out/pack_layout.json` | pitches, counts, gaps, fit check, class and part counts |
| `out/render.png` | classified view |
| `out/render_cells.png` | cell bank alone |
| `out/render_temperature.png` | temperature-coloured view |
| `out/render.html` | interactive browser view (13 MB) |
| `cache/geometry_*.pkl.gz` | tessellated meshes, keyed on a STEP fingerprint |

`cell_positions.csv` ordering is the contract with Task 2: `--temps` expects 408
values in exactly that order (row, then column, then bank).

## Notes carried into Task 2

- The temperature path was wired and tested with a placeholder gradient before the
  physics existed; `src/simulate.py` now supplies the real field.
- The cells run along $y$ and rows stack along $z$, so cross-flow is in the $x$–$z$
  plane. **Which of $\pm z$ is the inlet is not determined by the geometry** — the
  fan and duct were suppressed before export. Task 2 resolves the flow *direction*
  from the review (slide 100), but the inlet *end* is still worth confirming with
  the team; it does not change any number below, only which end of the render is hot.
- Cache keys include the tessellation tolerance and a STEP fingerprint, so
  re-exporting the CAD invalidates it automatically.

---

# Task 2 — airflow and thermal model

`src/params.py` (every input, with a source) → `src/flow.py` (the physics) →
`src/simulate.py` (run it, write results) → `src/plots.py` (figures).

## The result

Worst case from the design review: 40 °C inlet air, car stopped so the fans are
the only source of flow, 5200 W draw → 295.7 W of pack $I^2R$ loss.

| | |
|---|---|
| operating flow | **93.2 CFM** (2.64 m³/min), face velocity **1.375 m/s** |
| $V_{\max}$ in the bank | 13.4 m/s |
| system pressure drop | **1455 Pa** (148 mmH₂O) — **98% of it the cell bank** |
| $\mathrm{Re}_{\max}$ / $\mathrm{Nu}$ / $h$ | 14 370 / 91 / **137 W·m⁻²K⁻¹** |
| exhaust air | 46.0 °C (+5.9 °C) |
| hottest cell | **47.0 °C** |
| margin to the 60 °C limit | **+13.0 °C — PASS** |
| against the 45 °C design goal | **missed by 2.0 °C** |

Energy balance closes to 0.001%, $\mathrm{Re}$ stays inside the Žukauskas
correlation's $10^3$–$2\times10^5$ validity range across every row, and the fan
and system pressures agree to 0.000 Pa at the operating point (the solver
iterates the flow and the air state to consistency, since the system curve
depends on a density that depends on the flow).

## Why the operating point is the interesting part

The design review never solved for the flow. It took the fan's free-delivery
rating and scaled it by a 0.25 "airflow fudge" and an 0.85 duct efficiency to
get 1.3684 m/s, without checking whether the fans can push that much air against
the bank's resistance.

This model closes that loop — it intersects the digitized Delta P–Q curve with
the Žukauskas system curve — and lands at **1.375 m/s**, which is **0.5% from
their assumed value**. Two unrelated routes to the same number is a strong
mutual check, and it says their fudge factor was well chosen.

The fan-curve plot also shows *why* the system is so insensitive: at 1455 Pa the
fans sit on the flat, near-shut-off part of their curve.

## Validation against the review's own MATLAB study

Run at the review's imposed face velocities rather than at its own operating
point, this model reproduces their results within 11–22% across a 5× range of
velocity — from two independent derivations that share only the textbook.

| $v$ [m/s] | source | review | this model | ratio |
|---|---|---|---|---|
| 1.86 | slide 18 | 4.70 °C | 5.72 °C | 1.22 |
| 1.00 | slide 18 | 8.60 °C | 10.12 °C | 1.18 |
| 0.50 | slide 18 | 17.08 °C | 19.36 °C | 1.13 |
| 0.34 | slide 18 | 25.09 °C | 27.89 °C | 1.11 |
| 1.3684 | slide 105 | 5.85 °C | 7.02 °C | 1.20 |

This model runs consistently ~15% hotter, i.e. conservative. Part of that is
deliberate: it uses the **measured** 130 mm tube length where the review used
121 mm, and the friction factor is extrapolated below the tightest published
pitch curve in a direction that over-predicts $\Delta p$.

## Robustness

| case | flow | hottest cell | margin |
|---|---|---|---|
| design (3 fans, 295.7 W, 40 °C) | 93.2 CFM | 47.0 °C | +13.0 °C |
| higher load, 318.8 W (slide 17) | 93.2 CFM | 47.6 °C | +12.4 °C |
| San Ace 80 ×3 (spec'd fan) | 81.3 CFM | 48.0 °C | +12.0 °C |
| **one fan failed** | 90.8 CFM | 47.2 °C | +12.8 °C |
| **two fans failed** | 78.6 CFM | 48.2 °C | +11.8 °C |
| **50 °C ambient** | 94.5 CFM | 57.1 °C | **+2.9 °C** |

Two things fall out of this:

- **Fan redundancy is excellent.** Losing two of the three fans costs only 1.2 °C,
  because the fans are working near shut-off where the curve is flat — flow is
  set by the bank's resistance, not by how many fans are pushing.
- **Ambient temperature is the binding risk, not the heat load.** The pack
  tolerates ~800 W (2.7× design) before reaching 60 °C, but only ~10 °C of extra
  inlet air. The review's own worst case already flags 55 °C asphalt and
  crosswind, neither of which this 1-D model captures.

## What the CAD changed

The earlier `flow_model/` assumed an equilateral bank, $S_L = S_T = 20.6$ mm, and
set the Žukauskas friction correction $\chi = 1$ on that basis. The CAD gives
$S_L = 25.20$ mm, so $S_T/S_L = 0.813$ and the bank is not equilateral:

- $\chi$ becomes **1.207**, not 1. On its own that costs ~9% of the flow.
- $C_1 = 0.35(S_T/S_L)^{1/5}$ in the Nusselt correlation shifts with it.
- The minimum-flow plane is now unambiguously the **transverse** one — the
  transverse gap is 2.10 mm against a 17.6 mm diagonal gap — which the
  equilateral assumption could not have established.

The effect on the answer is small (0.6 °C), because the cell temperature is set
mostly by the air's enthalpy rise rather than by $h$. But it is the difference
between a result that rests on an assumption and one that rests on the geometry.

## Sensitivity to the assumed inputs

The only genuinely ASSUMED numbers are the duct and grille loss coefficients
($K = 3.5 + 1.4$), which the review does not give. Doubling them moves the
hottest cell by **0.2 °C** — they are 2% of the system resistance, so the result
does not depend on them.

## Outputs

| file | contents |
|---|---|
| `out/summary.png` | **all six figures on one sheet** |
| `out/fan_curve.png` | fan vs system curve, operating point marked |
| `out/thermal_march.png` | air and cell temperature row by row |
| `out/velocity_sweep.png` | performance vs flow, with the review's points |
| `out/sensitivity.png` | heat load, ambient, fan count |
| `out/results.json` | operating point, row march, sweeps, variants, validation |
| `out/cell_temperatures.csv` | all 408 cells with position and temperature |
| `out/cell_temperatures.json` | the array `render.py --temps` consumes |
| `out/render_temperature.png` | the 3D geometry coloured by the computed field |

## Assumptions and limits

- **Flow direction** is taken from the review (slide 100: $N = 204$ tubes,
  $N_T = 12$, $L = 130$ mm ⇒ 17 rows deep), not from the CAD — the fan and duct
  were suppressed before the STEP export, so the geometry alone cannot say which
  end is the inlet. Worth confirming with the team.
- The friction factor is extrapolated below Incropera Fig. 7.14's tightest
  published curve ($P_T = 1.25$) to this pack's $P_T = 1.114$. **This is the
  largest single uncertainty in the pressure drop.** It is conservative.
- The San Ace 80 row uses published endpoints with an assumed curve shape; it is
  context, not a sizing result.
- Heat is spread evenly over all cells; in reality the cells nearest the busbars
  run hotter. The model resolves temperature by row only, so all 24 cells in a
  row share a value.
- 1-D: uniform flow across the face, no crosswind, no recirculation, no radiative
  gain from the track — all of which the review's own worst case flags as real.
