# CalSol Excalibur — Battery Box Airflow & Thermal Model

A reduced-order airflow and thermal model of the CalSol Excalibur battery
enclosure, built from the actual CAD rather than from assumed dimensions, with an
interactive 3D explorer on top of it.

The pack is **408 LG INR18650 MJ1 cells** in a sealed aluminium box, cooled by
fans blowing through the cell bank. The question this repo answers: *at the worst
case the design review specifies, how hot does the hottest cell get, and does the
cooling actually work?*

**Short answer: yes, with 13 °C of margin — but the design goal is missed by 2 °C,
and one fan on each face would beat three fans on one.**

```bash
pip install -r requirements.txt
python src/app.py          # the interactive model, in your browser
python src/simulate.py     # the numbers, written to out/
```

---

## Contents

- [The headline result](#the-headline-result)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Part 1 — Geometry from CAD](#part-1--geometry-from-cad)
- [Part 2 — The physics](#part-2--the-physics)
- [Part 3 — Results](#part-3--results)
- [Part 4 — The interactive app](#part-4--the-interactive-app)
- [Which way does the air flow?](#which-way-does-the-air-flow)
- [Verification](#verification)
- [Parameter provenance](#parameter-provenance)
- [Limitations — what to trust least](#limitations--what-to-trust-least)
- [Running costs and hosting](#running-costs-and-hosting)
- [Roadmap](#roadmap)

---

## The headline result

Worst case from the design review: **40 °C inlet air**, car stopped so the fans
are the only source of flow, 5200 W draw → **295.7 W** of pack $I^2R$ loss,
3 × Delta GFB0812ES-E fans on one face.

| quantity | value |
|---|---|
| operating flow | **93.2 CFM** (2.64 m³/min), face velocity **1.375 m/s** |
| $V_{\max}$ in the bank | 13.4 m/s |
| system pressure drop | **1455 Pa** (148 mmH₂O) — **98 % of it the cell bank** |
| $\mathrm{Re}_{\max}$ / $\mathrm{Nu}$ / $h$ | 14 370 / 91 / **137 W·m⁻²K⁻¹** |
| exhaust air | 46.0 °C (+5.9 °C) |
| **hottest cell** | **47.0 °C** |
| margin to the 60 °C limit | **+13.0 °C — PASS** |
| against the 45 °C design goal | missed by 2.0 °C |

Three findings worth more than the headline number:

1. **The design review never solved for the flow.** It took the fan's
   free-delivery rating and scaled it by a 0.25 "airflow fudge" and an 0.85 duct
   efficiency to get 1.3684 m/s, without checking whether the fans can push that
   much air against the bank's resistance. This model closes the loop properly —
   intersecting the digitized fan curve with the system curve — and lands at
   **1.375 m/s, 0.5 % from their assumed value**. Two unrelated routes to the
   same number is a strong mutual check: their fudge factor was well chosen.

2. **More fans barely help; a second *face* does.** The cell bank is ~98 % of the
   system resistance, so the fans run at ~99 % of dead-head pressure where their
   curve has collapsed. Going 3 → 12 fans on one face buys **0.1 °C for 227 W**.
   But putting one fan on *each* face puts them in **series**, so their pressure
   rises add: **1 + 1 beats 3 + 0, cooler and at two-thirds the power.**

3. **Ambient temperature is the binding risk, not the heat load.** The pack
   tolerates ~800 W (2.7× design) before hitting 60 °C, but only ~10 °C of extra
   inlet air. Losing two of three fans costs just 1.2 °C.

---

## Quickstart

Python **3.10** (OCP wheels are version-sensitive).

```bash
git clone https://github.com/TheB2D/flowsimulationcalsol
cd flowsimulationcalsol

python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first geometry load tessellates the STEP and caches it — **~2.4 s** for the
enclosure, **~45 s** for the full assembly. Every run after that is a cache hit.
`cache/` is gitignored because it is derived data (136 MB of pickles).

```bash
# the interactive model — all three modes, in your browser
.venv/bin/python src/app.py
.venv/bin/python src/app.py --source box     # enclosure only, loads in ~2 s
.venv/bin/python src/app.py --prebuild       # warm the caches and exit

# the static pipeline
.venv/bin/python src/geometry.py    # read + classify + tessellate
.venv/bin/python src/inventory.py   # write out/*.csv and out/pack_layout.json
.venv/bin/python src/simulate.py    # flow + thermal, writes results and figures
.venv/bin/python src/render.py      # interactive geometry window

# colour the 3D render by the computed field
.venv/bin/python src/render.py --temperature --temps "$PWD/out/cell_temperatures.json"

# both flow directions, side by side
.venv/bin/python src/direction.py
.venv/bin/python src/direction.py --verify    # re-derive both from the CAD

# the UI regression suite
.venv/bin/python src/test_ui.py
```

**`--temps` needs an absolute path.** `render.py` falls back to a placeholder
gradient if the file is missing, printing `no temperature file given -- showing a
PLACEHOLDER gradient`.

| flag | applies to | meaning |
|---|---|---|
| `--source {legacy,box,full,pack}` | `app.py` | which STEP to load (default `full`) |
| `--tol MM` | `app.py`, `render.py` | tessellation tolerance (default 0.5) |
| `--port N` | `app.py` | server port |
| `--temperature`, `--temps FILE` | `render.py` | colour cells by a field |
| `--cells-only`, `--no-case`, `--light` | `render.py` | view variants |
| `--html`, `--png` | `render.py` | export instead of display |
| `--only {from_12,from_17}`, `--export`, `--verify` | `direction.py` | flow direction |

---

## Repository layout

```
├── BoxAssembly.step            the original export: box + pack (2804 solids)
├── fullbatterybox/             the newer CAD set
│   ├── Main Battery.step        box + pack + electronics (3374 solids) — app default
│   ├── Battery Box.step         enclosure alone (90 solids, true B-rep)
│   ├── Battery Pack.step        the 17-module pack
│   └── Battery Module.step
├── src/
│   ├── geometry.py             XCAF STEP reader, classification, mesh cache
│   ├── inventory.py            body/cell inventories → CSV + JSON
│   ├── openings.py             the two measured panel cutouts
│   ├── params.py               EVERY physical input, with units and a source
│   ├── flow.py                 operating point, air march, cell temperatures
│   ├── network.py              pressure-drop network, series/parallel fans
│   ├── placement.py            exhaustive Pareto search over fan arrangements
│   ├── transient.py            stochastic per-cell time march
│   ├── direction.py            both flow directions from one lattice
│   ├── fans.py                 digitized fan P–Q curves
│   ├── packscene.py            PyVista scene assembly, depth peeling
│   ├── camera.py               six named views, keyboard control
│   ├── render.py               static renders and HTML export
│   ├── plots.py, figures.py    matplotlib figure sheets
│   ├── equations.py            the 17 governing equations, rendered
│   ├── simulate.py             the batch run
│   ├── app.py                  the Trame application
│   └── test_ui.py              UI regression suite
├── out/                        figures, CSVs, results.json
├── writeup/report.pdf          the LaTeX report
├── DIRECTION.md                deep dive: the two flow directions
├── FULLBOX.md                  deep dive: the three app modes + bug log
└── CLAUDE.md                   original project brief
```

Three documents, by depth: **this README** is the overview, **`FULLBOX.md`** covers
the interactive model and carries a detailed engineering bug log, **`DIRECTION.md`**
covers the flow-direction ambiguity. **`writeup/report.pdf`** is the formal
write-up (Geometry → Governing equations → Operating conditions → Results →
Verification → Limitations → Conclusions).

---

## Part 1 — Geometry from CAD

### What the STEP file turned out to be

The obvious approach — `importStep(...).Solids()` — does not work here, for two
reasons. Current CadQuery returns a `Workplane`, so the call is `.val().Solids()`;
and more importantly **it throws away the assembly tree**. The files are read
instead through the **XCAF** document reader, which preserves the part name and
colour on every solid.

That matters because **classification keys off CAD part names, not bounding-box
guesswork** — and the names are what tie each body back to the BOM.

`BoxAssembly.step` is a **hybrid AP242 export**: 70 unique B-rep part bodies,
instanced by 3162 assembly occurrences into **2804 placed solids**. Tessellating
the unique parts once and reusing them is why the cold build is ~33 s rather than
tens of minutes.

| class | count | what it is |
|---|---:|---|
| `busbar` | 949 | busbars, interconnects, terminals, tabs |
| `structure` | 878 | divider boards, module spacers, tie rods, tape, covers |
| `cell` | 408 | LG INR18650 MJ1 |
| `other` | 389 | unclassified small parts |
| `fastener` | 173 | screws, washers |
| `case_plate` | 7 | the enclosure — 1 bottom + 2 back + 2 side + 2 top |

### Measured pack geometry

| quantity | value |
|---|---|
| cells | **408** = 17 rows × 12 across × 2 banks |
| arrangement | **staggered**, alternate rows offset by exactly $S_T/2 = 10.25$ mm |
| cell diameter $D$ | **18.40 mm** |
| cell length | **65.00 mm** |
| transverse pitch $S_T$ | **20.50 mm** ($S_T/D = 1.114$) |
| longitudinal pitch $S_L$ | **25.20 mm** ($S_L/D = 1.370$) |
| diagonal pitch $S_D$ | **27.20 mm** |
| minimum gap between cells | **2.10 mm** |
| tube length $L$ | 130.0 mm |
| frontal area | 319.8 cm² |
| $A_{\min}/A_{\text{face}}$ | 0.1024 |
| total cell wetted area | **1.5330 m²** |
| axes | rows along $z$, columns along $x$, cell axis along $y$ |

### Three measurement details that changed the answer

**Units are millimetres**, confirmed the way the brief asks: the cells measure
65.00 mm long and 18.40 mm across, which is an 18650 by definition.

**$D$ is measured off the cylindrical face, not the bounding box.** The bounding
box reads 19.92 mm because it includes the terminal button and the tab. The flow
model needs the can, since that is the cylinder the air actually sees — and the
difference is not cosmetic: it moves the inter-cell gap from 0.58 mm to 2.10 mm,
**a factor of 3.6 in the quantity that sets the pressure drop.**

**The bank is not equilateral.** Earlier work assumed $S_L = S_T = 20.6$ mm and
set the Žukauskas friction correction $\chi = 1$ on that basis. The CAD gives
$S_L = 25.20$ mm, so $S_T/S_L = 0.813$:

- $\chi$ becomes **1.207**, not 1 — on its own that costs ~9 % of the flow
- $C_1 = 0.35(S_T/S_L)^{1/5}$ in the Nusselt correlation shifts with it
- the minimum-flow plane is now unambiguously the **transverse** one (2.10 mm
  against a 17.6 mm diagonal gap), which the equilateral assumption could not
  have established

The effect on the final temperature is small (0.6 °C), because cell temperature
is set mostly by the air's enthalpy rise rather than by $h$. But it is the
difference between a result resting on an assumption and one resting on the
geometry.

### Pack placement check

The brief warns that the pack was positioned by hand in Onshape with no mates.
Checked: **all 408 cells sit inside the enclosure**, with 217/178 mm clearance in
$x$, 37/11 mm in $y$ and 49/28 mm in $z$. No interference.

51 non-cell bodies (module spacers, divider boards) report as outside the box,
but that is a naming artifact of this export — several multi-body parts give all
their solids one part name — not a placement problem. `size_outliers()` in
`src/inventory.py` reports these and confirms **none of them are cells**, which
is the check that matters.

### The two panel cutouts

| CAD part | colour in the render | measured cutout |
|---|---|---|
| **`Front Plate-1`** | blue | 251.3 × 121.0 mm, clear |
| **`Rear Plate-1`** | green | 244.0 × 124.0 mm, **porosity 0.886** (truss) |

They sit **590 mm apart on the $z$ axis** with the cell block between them. Each
opening fits **exactly 3 × 80 mm fans**, which independently confirms the design
review's "3 across the pack".

**Panels are identified by name, never by colour.** OCCT returns *linear* RGB, so
the blue panel reads back as `#000BA4` rather than the `#023DD2` stored in the
STEP. Colour matching would have silently picked the wrong parts.

---

## Part 2 — The physics

`src/params.py` (every input, with a source) → `src/flow.py` (the physics) →
`src/simulate.py` (run it) → `src/plots.py` (figures).

### 1. Fan operating point

The flow is where the fans' pressure rise equals the system's pressure demand:

$$\Delta p_{\text{fan}}(Q) = \Delta p_{\text{sys}}(Q)$$

The fan side is the **digitized Delta GFB0812ES-E P–Q curve**; $n$ fans on one
face are in parallel ($Q$ adds), and fans on opposite faces are in **series**
($\Delta p$ adds):

$$\sum_i \Delta p_i(Q) = \Delta p_{\text{sys}}(Q)$$

The system side is the cell bank plus duct and grille losses. The bank uses the
Žukauskas staggered tube-bank form:

$$\Delta p_{\text{bank}} = N_L\,\chi\,f\,\frac{\rho V_{\max}^2}{2},
\qquad
V_{\max} = v_{\text{face}}\,\frac{A_{\text{face}}}{A_{\min}}$$

Solved with Brent's method. The solver iterates the flow and the air state to
consistency, because the system curve depends on a density that depends on the
flow — at the operating point the two pressures agree to **0.000 Pa**.

### 2. Heat generation

$$\dot Q_{\text{cell}} = I_{\text{cell}}^2 R_{\text{int}},
\qquad I_{\text{cell}} = \frac{I}{N_p}$$

At 34S12P, 61.18 A pack current and $R_{\text{int}} = 28$ mΩ this gives
**295.7 W** over 408 cells.

### 3. Air temperature march

Row by row along the flow path, each row dumping its heat into the air:

$$\dot m\,c_p\,\Delta T_{\text{air}} = \sum \dot Q_{\text{cell}}$$

### 4. Cell temperature

$$T_{\text{cell}} = T_{\text{air,local}} + \frac{\dot Q_{\text{cell}}}{h A_{\text{cell}}}$$

with $h$ from the Žukauskas correlation for staggered banks, $\mathrm{Re}$ based
on $V_{\max}$ in the cell gaps and $D = 18.4$ mm:

$$\mathrm{Nu} = C_1 C_2 \,\mathrm{Re}_{\max}^{m}\,\mathrm{Pr}^{0.36}
\left(\frac{\mathrm{Pr}}{\mathrm{Pr}_s}\right)^{1/4}$$

### 5. Transient (optional)

Per-cell lumped nodes integrated forward:

$$C_i\frac{dT_i}{dt} = \dot q_i - h_iA_i\,(T_i - T_{\text{air}})
+ \sum_j G_{ij}(T_j-T_i) + \underbrace{\sigma_\eta\xi_i(t)}_{\text{optional, not physics}}$$

| quantity | value |
|---|---|
| cell heat capacity $C=mc_p$ | 50.4 J/K |
| time constant $C/hA$ | **49.1 s** |
| explicit-Euler stability limit | 35.6 s (running at $dt$ = 0.5 s, a 71× margin) |
| cost | **~1 µs/step** → a 400 s transient solves in ~9 ms |

Fast enough to run **live** rather than replaying precomputed frames.

All 17 governing equations are rendered in-app from `src/equations.py`.

---

## Part 3 — Results

### Validation against the review's own MATLAB study

Run at the review's *imposed* face velocities rather than at its own operating
point, this model reproduces their results within **11–22 %** across a 5× range
of velocity — from two independent derivations that share only the textbook.

| $v$ [m/s] | source | review | this model | ratio |
|---|---|---|---|---|
| 1.86 | slide 18 | 4.70 °C | 5.72 °C | 1.22 |
| 1.00 | slide 18 | 8.60 °C | 10.12 °C | 1.18 |
| 0.50 | slide 18 | 17.08 °C | 19.36 °C | 1.13 |
| 0.34 | slide 18 | 25.09 °C | 27.89 °C | 1.11 |
| 1.3684 | slide 105 | 5.85 °C | 7.02 °C | 1.20 |

Consistently ~15 % hotter, i.e. **conservative**. Part of that is deliberate: it
uses the **measured** 130 mm tube length where the review used 121 mm, and the
friction factor is extrapolated in a direction that over-predicts $\Delta p$.

### Robustness

| case | flow | hottest cell | margin |
|---|---|---|---|
| design (3 fans, 295.7 W, 40 °C) | 93.2 CFM | 47.0 °C | +13.0 °C |
| higher load, 318.8 W (slide 17) | 93.2 CFM | 47.6 °C | +12.4 °C |
| San Ace 80 ×3 (spec'd fan) | 81.3 CFM | 48.0 °C | +12.0 °C |
| **one fan failed** | 90.8 CFM | 47.2 °C | +12.8 °C |
| **two fans failed** | 78.6 CFM | 48.2 °C | +11.8 °C |
| **50 °C ambient** | 94.5 CFM | 57.1 °C | **+2.9 °C** |

**Fan redundancy is excellent** — losing two of three fans costs only 1.2 °C,
because the fans work near shut-off where the curve is flat; flow is set by the
bank's resistance, not by how many fans push. **Ambient is the binding risk**:
~800 W of heat is tolerable, but only ~10 °C of extra inlet air. The review's own
worst case already flags 55 °C asphalt and crosswind, neither of which this 1-D
model captures.

### Sensitivity to the assumed inputs

The only genuinely **assumed** numbers are the duct and grille loss coefficients
($K = 3.5 + 1.4$), which the review does not give. Doubling them moves the
hottest cell by **0.2 °C** — they are 2 % of the system resistance in the baseline
direction, so the result does not depend on them. (This is *not* true in the
`from_17` direction — see below.)

### Extra heat sources

At 61.2 A: contactor $I^2R$ 1.5 W, coil hold 16 W, PCBs 15 W → **≈32.5 W, 11 % of
the 296 W cell load** — all **assumed**, none in the design review.

Measured placement decides how they enter: the 4 Gigavacs sit at $z$ = −583…−526
(at the green plate) and both boards at $y$ = 196…198 (on the shelf), so **both
are outside the cell air path**. They raise the *exhaust* by ~0.65 K and leave
peak cell temperature untouched. Adding them to the cell load would have inflated
every cell temperature by ~11 %.

### Output files

| file | contents |
|---|---|
| `out/summary.png` | **all six figures on one sheet** |
| `out/fan_curve.png` | fan vs system curve, operating point marked |
| `out/thermal_march.png` | air and cell temperature row by row |
| `out/velocity_sweep.png` | performance vs flow, with the review's points |
| `out/sensitivity.png` | heat load, ambient, fan count |
| `out/results.json` | operating point, row march, sweeps, variants, validation |
| `out/body_inventory.csv` | 2804 rows — index, name, class, volume, centroid, bbox |
| `out/cell_positions.csv` | 408 rows — `cell_id`, `(row, col, bank)`, position, $D$ |
| `out/cell_temperatures.csv` | all 408 cells with position and temperature |
| `out/cell_temperatures.json` | the array `render.py --temps` consumes |
| `out/pack_layout.json` | pitches, counts, gaps, fit check, class counts |
| `out/render.html` | standalone interactive browser view (13 MB) |
| `out/render*.png` | classified, cells-only and temperature-coloured views |

`cell_positions.csv` ordering is the contract between the geometry and the
physics: `--temps` expects 408 values in exactly that order (row, then column,
then bank).

---

## Part 4 — The interactive app

```bash
.venv/bin/python src/app.py
```

The 3D view and the figure sheet sit side by side; the drawer carries the physics
sliders, appearance toggles, time controls and the governing equations. Full
detail and the engineering bug log are in **`FULLBOX.md`**.

### Mode 1 — optimal fan placement

Exhaustively enumerates all **30 buildable arrangements** over the two measured
cutouts (each face ∈ {off, in, out} × 0–3 fans, requiring a path in *and* out)
and solves each. Exhaustive rather than an optimiser because the space is tiny
and each solve is milliseconds — which also makes it provably optimal within the
space.

**The result is a trade-off, not a single winner**, so the search reports the
Pareto front:

| arrangement | flow | hottest cell | fan power |
|---|---|---|---|
| *current: 3 one-sided* | *93.2 CFM* | *47.02 °C* | *75.6 W* |
| 1 green in (vent through blue) | 84.1 CFM | 47.72 °C | 25.2 W |
| **1 blue in + 1 green out** | **107.1 CFM** | **46.18 °C** | **50.4 W** |
| 2 blue in + 2 green out | 127.1 CFM | 45.28 °C | 100.8 W |
| **3 blue in + 3 green out** | **131.6 CFM** | **45.12 °C** | 151.2 W |

**One fan on each face beats three on one face — cooler *and* at two-thirds the
power.** Meanwhile 3 → 6 → 12 fans on a single face gives 93.2 → 94.4 → 94.7 CFM:
**0.1 °C for 227 W**. The system is *pressure*-starved, not flow-starved, which is
exactly what series staging fixes.

### Mode 2 — two-sided blast

Push at one face and pull at the other and the two fan sets are in series, so
their pressure rises add against one system curve. 3+3 reaches **131.6 CFM and
45.12 °C** — a **1.9 °C gain for no change to the geometry**, very nearly meeting
the 45 °C design goal.

**Both panels blowing in is refused.** The box is sealed apart from these two
cutouts, so continuity $\sum Q_{\text{in}} = \sum Q_{\text{out}}$ has **no
solution** for $Q>0$ — the box pressurises until one face reverses. The model
raises `InfeasibleTopology` with the explanation rather than returning a
plausible-looking number. A face with no fan is still a vent, which is why the
one-sided case remains valid.

### Mode 3 — stochastic transient

Per-cell time march with play / pause / step / scrub. Two kinds of randomness,
each labelled:

- **Property scatter (physical)** — per-cell $R_{\text{int}}$ and convective
  coupling drawn from a seeded distribution, so a run is bit-reproducible.
- **Additive OU noise (visual)** — off by default, Ornstein–Uhlenbeck rather than
  white so it reads as drift rather than a rendering bug.

Two honesty points are built into the UI rather than papered over:

**The scatter is small.** At $\sigma_R$ = 5 % the per-cell spread is ~60 mK
against a 6 K rise along the flow — about 1 % of the visible gradient. Rather
than quietly inflating $\sigma$, there is a separate **scatter gain** slider
defaulting to 1.0 and labelled *"> 1 is NOT physical"*, plus a **departure from
row mean** view showing the scatter on its own scale at true magnitude.

**The lumped assumption is formally violated.** $\mathrm{Bi}=h(D/4)/k_r$ is
**0.6–3.1** here, well above 0.1, so a node is a **volume-mean** temperature, not
a surface temperature. The centre-to-surface rise is 0.9–4.5 K — larger than the
entire scatter effect. The **surface T** toggle adds the radial correction
$\dot q/(4\pi kL)$ before any comparison against the 60 °C *surface* limit.

### Camera and appearance

Hover the pointer over the 3D view and it takes focus — no click first.

| input | orbit mode (default) | pan mode |
|---|---|---|
| arrow keys | yaw / pitch about the pack | slide sideways and up/down |
| shift + arrows | 4× coarser steps | 4× coarser steps |
| Q / E | roll | roll |
| + / − | dolly in / out | dolly in / out |
| R | reset to the default view | — |
| F | side-on flow view | — |
| drag / scroll | orbit / zoom | pan / zoom |

Six named views: **iso**, **flow** (side-on across the 590 mm axis, the view that
shows the row-to-row gradient), **inlet** and **exhaust** (head-on at the two
cutouts), **top**, **side**. Pitch is clamped to ±85° so the camera never passes
a pole and inverts the controls.

Panels, electronics shelf, PCBs and internal structure start **translucent** so
the cells read; one switch makes them solid. **Depth peeling** is enabled so
transparency is correct at any opacity. The colour scale is **pinned** (default
40–50 °C) rather than auto-scaled per frame, so the three modes stay visually
comparable.

---

## Which way does the air flow?

**This is not settled by the CAD** — the fans and ducts were suppressed before
the STEP export, so nothing in the geometry marks an inlet. Rather than pick one,
`src/direction.py` builds **both** and lets you toggle. Full treatment in
**`DIRECTION.md`**.

The 408 cells are one staggered lattice; a staggered bank is counted by the
successive **offset planes** the air crosses. Applying that consistently to the
same lattice, viewed two ways:

| | **from the 12** (flow ∥ $z$) | **from the 17** (flow ∥ $x$) |
|---|---|---|
| $S_T$ across the flow | 20.50 mm | **50.40 mm** |
| $S_L$ plane to plane | 25.20 mm | 10.25 mm |
| $N_L$ planes crossed | 17 | 24 |
| minimum gap | **2.10 mm** (transverse) | **17.61 mm** (diagonal) |
| $A_{\min}/A_{\text{face}}$ | 0.1024 | 0.3494 |
| flow | 93.2 CFM | **359.7 CFM** (3.86×) |
| $V_{\max}$ | 13.43 m/s | 8.72 m/s (0.65×) |
| $\Delta p$ bank | 1422 Pa | 234 Pa (0.16×) |
| $h$ | 136.7 W·m⁻²K⁻¹ | 126.9 W·m⁻²K⁻¹ |
| air rise | 5.95 °C | 1.54 °C |
| **hottest cell** | **47.0 °C** | **43.0 °C** |
| 45 °C goal | −2.0 °C (missed) | +2.0 °C (met) |

The "from the 17" column is the one that is easy to get wrong. Viewed along $x$
the lattice does **not** present 12 planes of 34 cells at the 25.2 mm row pitch.
It resolves into **24 planes at 10.25 mm**, each holding 8 or 9 cells at **50.4 mm**
pitch — so the transverse pitch that direction is 50.4 mm, not 25.2 mm, and the
minimum-area plane switches from transverse to diagonal.

Both descriptions recover the invariants they must, which is the check that they
are the same lattice: **408 cells**, **1.5330 m² of wetted area**, $S_D = 27.20$ mm.
`--verify` re-derives all nine checks from `out/cell_positions.csv`.

**Why the difference is so large:** turning the flow 90° widens the transverse
pitch from 20.5 mm to 50.4 mm, opening $A_{\min}/A_{\text{face}}$ by 3.4× and the
flow front by 1.74×. Against a nearly-flat fan curve the bank resists 6× less and
passes 3.9× the air. Note the trade — $h$ actually *drops* (127 vs 137) because
$V_{\max}$ is lower. **The win is not better heat transfer per cell; it is the
3.9× mass flow**, cutting the air rise from 5.95 °C to 1.54 °C.

Both directions pass the 60 °C limit; only "from the 17" meets the 45 °C goal.

> **Caveat:** in "from the 17" the duct and grille losses (~495 Pa) now *exceed*
> the bank (234 Pa), because the same fixed fan area passes 3.9× the flow and
> those losses go as $V_{\text{fan}}^2$. The assumed $K = 3.5 + 1.4$ was
> negligible in the baseline direction; here it is half the total, so **revisit it
> before trusting the ~360 CFM figure.**

---

## Verification

| check | result |
|---|---|
| fan and system pressure at the operating point | agree to **0.000 Pa** |
| global energy balance (steady) | closes to **0.001 %** |
| $\mathrm{Re}$ inside Žukauskas validity ($10^3$–$2\times10^5$) | every row |
| `network.py` reproduces `flow.py` | **exact, 0.0e+00 CFM difference** |
| `direction.py` `from_12` reproduces `flow.py` | 3 parts in $10^9$ (Brent tol) |
| transient → analytic steady state | max error **2.8e-12 K** |
| exponential approach at $t=\tau$ | 0.6336 vs $1-1/e$ = 0.6321 (0.15 %) |
| transient energy balance | **1.1e-15** |
| both-panels-in topology | raises `InfeasibleTopology` |
| lattice invariants, both directions | 9/9 checks pass |
| all 17 equations render offline | 17/17 |
| legacy `BoxAssembly.step` classification | unchanged (7 panels, 408 cells) |
| new assembly classification | 408 cells; all 10 `case_plate` genuine |
| `out/summary.png` | byte-identical across refactors |

`src/test_ui.py` is a real regression suite, not a smoke test: it builds the
actual UI, asserts `html.count("html-error") == 0`, checks all 26 control
bindings and 9 markup elements are present, verifies the mode toggle has exactly
3 buttons rather than 6, drives `animate()` for real and asserts simulated time
advances, checks every 80 mm fan frame fits inside its cutout, and **fails if the
sidebar exceeds 1 MB/s** of websocket traffic.

That last one exists because playback originally froze the app — the figure sheet
was being re-encoded on every animation frame at 3.26 MB/s, saturating the
websocket. See `FULLBOX.md` for that investigation and three silent-failure
Trame traps worth knowing about.

> **The general lesson, recorded because it cost real time:** "the server started
> and nothing crashed" is a much weaker check than it feels like when the
> deliverable is a UI. Verify the widgets are actually in the rendered output and
> that interactions actually do something; treat any warning during UI
> construction as a failure to be explained rather than noise to be dismissed.

---

## Parameter provenance

Every physical input lives in `src/params.py` with units and a source, tagged by
how much it can be trusted. `MEASURED` beats `REVIEW` beats `DATASHEET` beats
`ASSUMED`.

| parameter | value | source | note |
|---|---|---|---|
| cell diameter $D$ | 18.40 mm | **MEASURED** | cylindrical face in the STEP |
| cell length | 65.00 mm | **MEASURED** | STEP bounding box |
| rows × cols × banks | 17 × 12 × 2 | **MEASURED** | STEP cell lattice |
| $S_T$ | 20.50 mm | **MEASURED** | STEP, within-row pitch |
| $S_L$ | 25.20 mm | **MEASURED** | STEP, row-to-row pitch |
| stagger | $S_T/2$ = 10.25 mm | **MEASURED** | STEP; review p14 confirms intent |
| panel cutouts | 251.3 × 121.0, 244.0 × 124.0 mm | **MEASURED** | inner wires of planar faces |
| flow direction | across the 17 rows | REVIEW | p100: $N$=204, $N_T$=12, $L$=130 mm |
| pack config | 34S12P | REVIEW | p99 |
| cell resistance | 28 mΩ | REVIEW | p99 (LG MJ1) |
| heat load | 295.69 W | REVIEW | p99, at 5200 W draw / 61.18 A |
| heat load (alt) | 318.84 W | REVIEW | p17, at 5400 W draw / 63.53 A |
| inlet air | 40 °C | REVIEW | p15/p98 worst case |
| cell limit | 60 °C | REVIEW | p100 ($T_s$); p15 design goal is 45 °C |
| fan | 3 × Delta GFB0812ES-E | DATASHEET | P–Q curve, review p104 |
| fan shut-off | 150 mmH₂O | DATASHEET | review p104, 12 V curve |
| duct + grille $K$ | 3.5 + 1.4 | **ASSUMED** | not given in the review |
| duct face area | 246 × 130 mm | **ASSUMED** | = bank frontal area; ducting suppressed before export |
| contact conductance | 0.02 W/K | **ASSUMED** | the weakest value in the model |
| electronics heat | ≈32.5 W | **ASSUMED** | contactors + PCBs, outside the air path |

---

## Limitations — what to trust least

Ranked by how much they could move the answer.

1. **The friction-factor extrapolation dominates everything.** Bank $\Delta p$ is
   ~98 % of the total, and it rests on extrapolating Incropera Fig. 7.14's
   staggered friction curve below its tightest published pitch ($P_T = 1.25$) to
   this pack's $P_T = 1.114$. **This single extrapolation sets every number in
   all three modes.** It is conservative, but it is an extrapolation.

2. **Which direction the air flows is unresolved** — see above. It is worth
   ±4 °C on the peak cell, and it is a question for the team, not for the CAD.

3. **The duct/grille $K$ values are assumed.** Negligible in the baseline
   direction (2 % of resistance, 0.2 °C), but **half the total resistance** in the
   `from_17` direction.

4. **`contact_conductance` = 0.02 W/K is the weakest assumed value** and the
   first thing worth measuring.

5. **No CFD or experimental validation exists for the push–pull case.** Series
   composition assumes the two fan sets do not aerodynamically interfere —
   reasonable at 590 mm apart with the bank between them, but unverified.

6. **Heat is spread evenly over all cells.** In reality the cells nearest the
   busbars run hotter. The steady model resolves temperature by row only, so all
   24 cells in a row share a value.

7. **It is a 1-D row march.** Uniform flow across the face, no bypass around the
   bank, no recirculation, no crosswind, no radiative gain from the track — all
   of which the review's own worst case flags as real. Edge bypass would hurt
   `from_17` more than `from_12`, because its bank resistance is lower relative
   to any leak path.

8. **The lumped transient nodes have $\mathrm{Bi} \gg 0.1$** (0.6–3.1), so node
   temperatures are volume means; use the *surface T* toggle before comparing
   against the 60 °C surface limit.

The numbers in this README were taken at one point in time. Re-run
`src/simulate.py` and `src/direction.py --export` after any parameter edit rather
than trusting the tables.

---

## Running costs and hosting

Measured peak resident memory on this codebase, loading the geometry and building
the PyVista scene:

| source | solids | peak RSS |
|---|---:|---:|
| `box` (enclosure only) | 90 | **461 MB** |
| `full` (the app default) | 3374 | **1354 MB** |

Practical consequences:

- **Budget ~2 GB of RAM.** Every 512 MB free hosting tier is out.
- **`app.py` cannot deploy to serverless platforms** (Vercel, Netlify, Lambda).
  Trame's `PyVistaRemoteView` renders frames server-side with an OpenGL context
  and streams them over a persistent websocket — it needs a long-lived process,
  and the `vtk` + `OCP` dependencies alone are ~580 MB on disk.
- **`out/render.html` *is* fully static** and can go on any static host, if an
  interactive 3D view without the physics controls is enough.
- For the live app, use a container host with ≥2 GB (Hugging Face Spaces' free
  tier is 16 GB), or a **Cloudflare Tunnel** from a local machine.

Two notes for anyone containerising this:

- **The cache key includes mtime.** `_step_fingerprint` (`src/geometry.py:568`)
  keys on `size_mtime_headhash`, and `git clone` sets mtime to checkout time —
  so a committed cache would still miss and cold-build. Drop mtime from the key,
  or fall back to any `geometry_{stem}_*_tol{tol}.pkl.gz` match.
- **`cadquery` itself is not imported anywhere** — only `OCP.*` directly. Install
  `cadquery-ocp` without `cadquery` and its `casadi` (158 MB), `pymupdf` (57 MB)
  and `ezdxf` (20 MB) dependencies drop out of the image.

---

## Roadmap

The reduced-order model is the calibration target for CFD, not a replacement.

- **OpenFOAM `chtMultiRegionFoam`** (conjugate heat transfer), via Docker and
  driven from Python. Meshing the 2.10 mm gaps between cells is the hard part.
- **Gmsh + FEniCSx** for detailed conduction in the pack, using $h$ from this
  model as the boundary condition:
  $$-k\,\frac{\partial T}{\partial n} = h\,(T - T_\infty)$$
- **Homogenised modules** as a cheaper intermediate: each module an anisotropic
  block with $k_\parallel \neq k_\perp$ (along vs. across the cell axis) and a
  volumetric heat source.
- Measure the two things the model cannot settle on its own: the **actual flow
  direction** and the **duct/grille loss coefficients**.

> **SimScale** (cloud CFD with an Onshape integration) is a tempting alternative,
> but free-plan projects are **public**. Check with the team before uploading
> CalSol CAD.

---

## Conventions

- The STEP files are the source of truth. Never modify the original SolidWorks
  files or the PDFs.
- Every physical parameter lives in `src/params.py`, with units and a source.
- `cache/` is derived data and is gitignored; it rebuilds itself.
- Ask before assuming anything about the airflow path, fan count, or electrical
  configuration — the three things the CAD does not settle.
