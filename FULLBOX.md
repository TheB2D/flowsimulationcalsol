# Full battery box — three-mode interactive model

```bash
.venv/bin/python src/app.py                 # opens in your browser
.venv/bin/python src/app.py --source box    # enclosure only, loads faster
.venv/bin/python src/app.py --prebuild      # warm the caches and exit
```

The 3D view and the research-paper figure sheet sit side by side; the drawer
carries the physics sliders, the appearance toggles, the time controls and the
governing equations.

---

## The two panels you pointed at

| your words | CAD part | measured cutout |
|---|---|---|
| green panel with rectangular hole | **`Rear Plate-1`** | 244.0 × 124.0 mm, **porosity 0.886** (purple truss) |
| blue panel with rectangular hole | **`Front Plate-1`** | 251.3 × 121.0 mm, clear |

They sit **590 mm apart on the z axis** with the cell block between them. Each
opening fits **exactly 3 × 80 mm fans**, which independently confirms the design
review's "3 across the pack".

**Panels are identified by NAME, never by colour.** OCCT returns *linear* RGB, so
the blue panel reads back as `#000BA4` rather than the `#023DD2` stored in the
STEP. Colour matching would have silently picked the wrong parts.

---

## Mode 1 — optimal fan placement

Exhaustively enumerates all **30 buildable arrangements** over the two measured
cutouts (each face ∈ {off, in, out} × 0–3 fans, requiring a path in and out) and
solves each. Exhaustive rather than an optimiser because the space is tiny and
each solve is milliseconds, which also makes it provably optimal within the space.

**The result is a trade-off, not a single winner**, and the search reports the
Pareto front because the objective is nearly flat in fan count:

| arrangement | flow | hottest cell | fan power |
|---|---|---|---|
| 1 green in (vent through blue) | 84.1 CFM | 47.72 °C | 25.2 W |
| **1 blue in + 1 green out** | **107.1 CFM** | **46.18 °C** | **50.4 W** |
| 2 blue in + 2 green out | 127.1 CFM | 45.28 °C | 100.8 W |
| **3 blue in + 3 green out** | **131.6 CFM** | **45.12 °C** | 151.2 W |

Against the current one-sided 3-fan design (93.2 CFM, 47.02 °C, 75.6 W):
**one fan on each face beats three on one face — cooler *and* at two-thirds the
power.**

### Why more fans barely help

| fans, one face | flow | hottest cell | power |
|---|---|---|---|
| 3 | 93.2 CFM | 47.02 °C | 75.6 W |
| 6 | 94.4 CFM | 46.94 °C | 151.2 W |
| 12 | 94.7 CFM | 46.92 °C | 302.4 W |

3 → 12 fans buys **0.1 °C for 227 W**. The cell bank is **~98% of the system
resistance** and the fans run at ~99% of dead-head pressure, so parallel fans add
capacity to a curve that has already collapsed. The system is *pressure*-starved,
not flow-starved — which is exactly what series staging fixes.

---

## Mode 2 — two-sided blast

Fans on the blue panel and the green panel at once. Push at one and pull at the
other and the two sets are **in series**, so their pressure rises **add** against
one system curve:

$$\sum_i \Delta p_i(Q) = \Delta p_{\rm sys}(Q)$$

That is the whole result: 3+3 reaches **131.6 CFM and 45.12 °C**, a **1.9 °C
gain** over the current arrangement for no change to the geometry, and it very
nearly meets the 45 °C design goal.

### Both panels blowing IN is refused

The box is sealed apart from these two cutouts, so continuity
$\sum Q_{\rm in} = \sum Q_{\rm out}$ has **no solution** for $Q>0$ — the box
pressurises until one face reverses. The model raises `InfeasibleTopology` with
the explanation rather than returning a plausible-looking number. A face with no
fan is still a vent, which is why the one-sided case remains valid.

---

## Mode 3 — stochastic transient

Per-cell lumped nodes integrated forward, with play / pause / step / scrub:

$$C_i\frac{dT_i}{dt} = \dot q_i - h_iA_i\,(T_i - T_{\rm air}) + \sum_j G_{ij}(T_j-T_i) + \underbrace{\sigma_\eta\xi_i(t)}_{\text{optional, not physics}}$$

Both kinds of randomness you asked for, each labelled:

- **Property scatter (physical)** — per-cell $R_{\rm int}$ and convective coupling
  drawn from a seeded distribution, so a run is bit-reproducible.
- **Additive OU noise (visual)** — off by default, Ornstein–Uhlenbeck rather than
  white so it looks like drift rather than a rendering bug.

| quantity | value |
|---|---|
| cell heat capacity $C=mc_p$ | 50.4 J/K |
| time constant $C/hA$ | **49.1 s** |
| explicit-Euler stability limit | 35.6 s (running at dt = 0.5 s, a 71× margin) |
| cost | **~1 µs/step** → a 400 s transient solves in ~9 ms |

So the transient runs **live** rather than replaying precomputed frames.

### Two honesty points built into the UI

**The scatter is small.** At σ_R = 5% the per-cell spread is ~60 mK against a 6 K
rise along the flow — about 1% of the visible gradient. Rather than quietly
inflating σ, there is a separate **`scatter gain`** slider defaulting to 1.0 and
labelled *"> 1 is NOT physical"*, plus a **"departure from row mean"** view that
shows the scatter on its own colour scale at true magnitude.

**The lumped assumption is formally violated.** $\mathrm{Bi}=h(D/4)/k_r$ is
**0.6–3.1** here, well above 0.1, so a node is a **volume-mean** temperature, not
a surface temperature. The centre-to-surface rise is 0.9–4.5 K — larger than the
entire scatter effect. The *"surface T"* toggle adds the radial correction
$\dot q/(4\pi kL)$ before any comparison against the 60 °C surface limit.

---

### Playback froze the app: the sidebar was saturating the websocket

Playing the transient while panning killed the connection — 1392
`Cannot write to closing transport` errors in one session. The server stayed
up; the websocket did not.

Profiling the per-frame cost made the cause obvious:

| step | cost |
|---|---|
| physics step | 0.01 ms |
| `push_field` (3D scalars) | 0.18 ms |
| `update_metrics` | 0.01 ms |
| **`refresh_sidebar`** | **~50 ms, ~200 KB** |

The sheet was being re-encoded on **every** animation frame: at the old 60 ms
loop that is 16.7 PNGs/s = **3.26 MB/s**, on top of the 3D view's own JPEG
stream. Panning added interaction frames to the same queue, the browser could
not drain it, and the transport closed.

Three fixes:

* **Throttle the sidebar** to `SIDEBAR_MIN_INTERVAL = 0.5 s`, independent of the
  3D frame rate. A time-series plot does not need 16 updates a second. Lowering
  dpi alone was not enough — the ~50 ms is matplotlib's *draw*, not the raster —
  though playback does now encode at `PLAY_DPI = 72` (~120 KB vs ~200 KB).
* **Budget the frame.** The loop sleeps the remainder of `FRAME_INTERVAL`
  rather than a fixed delay, and never less than 10 ms, so the event loop is
  never starved.
* **Rate-limit camera pushes** to ~30 fps while playing. The camera always
  moves; only the frame push is capped, so held arrow keys stay smooth without
  doubling the stream.

Measured, at the worst case of `speed = 1`:

| | before | after |
|---|---|---|
| sidebar encodes | 16.7 /s | **2.1 /s** |
| sidebar bandwidth | 3.26 MB/s | **0.27 MB/s** (12x less) |
| 201 key events while playing | 201 pushes | **109 pushes**, bounded |

Playback is also *faster* now (60 s simulated per 0.6 s wall, against 35 s
before) because the loop is no longer blocked re-encoding PNGs.

`src/test_ui.py` drives real playback and fails if the sidebar exceeds
1 MB/s.

### Fan placement: use the cutout centre, not the panel centroid

Fans were first drawn at each panel's **centroid**, which put them ~80 mm too
high — floating above the opening, up level with the contactors.

The plates carry more material above the cutout than below, so the centroid and
the hole are nowhere near each other:

| panel | centroid y | **cutout centre y** | error |
|---|---|---|---|
| blue `Front Plate-1` | 152.7 mm | **78.4 mm** | 74 mm high |
| green `Rear Plate-1` | 156.9 mm | **73.1 mm** | 84 mm high |

`Opening` now carries `centre_x` / `centre_y`, measured from the inner wires of
the plate's planar faces (front: x −125.6…125.6, y 17.9…138.9; rear: x
−122.0…122.0, y 11.1…135.2). `src/test_ui.py` checks that every 80 mm frame
fits entirely within its cutout, not merely that the centre point is nearby.

### Team logo

The CalSol logo sits as a watermark in the **top-right** of the 3D view, at 85%
opacity. Source artwork is `fullbatterybox/CalSol_Logo_White.png.webp`,
converted and trimmed to `assets/calsol_logo.png` (774 x 240, VTK does not read
WebP).

Two details that need care:

* `add_logo_widget` takes **independent** width and height fractions, so the
  height is derived from both the artwork's 3.23:1 aspect and the window's --
  passing a square size stretches the logo badly.
* Bottom-right is the obvious corner but the wrong one: the horizontal scalar
  bar spans the full width there and the logo lands on its upper end. Top-left
  holds the scene title and bottom-left the orientation axes, so **top-right is
  the only free corner**.

`src/test_ui.py` checks the asset exists, that the drawn aspect matches the
artwork to within 0.02, and that the logo is not low enough to collide with the
scalar bar. A missing asset degrades gracefully -- `add_logo` returns False and
the scene renders without it.

## 3D camera controls

The **Camera / view** panel in the drawer, plus keyboard control over the 3D
view. Hover the pointer over the view and it takes focus automatically — no
click first.

| input | orbit mode (default) | pan mode |
|---|---|---|
| arrow keys | yaw / pitch about the pack | slide the view sideways and up/down |
| shift + arrows | 4x coarser steps | 4x coarser steps |
| Q / E | roll | roll |
| + / − | dolly in / out | dolly in / out |
| R | reset to the default view | — |
| F | side-on flow view | — |
| drag | orbit | pan |
| scroll | zoom | zoom |

The **orbit / pan** toggle changes what the arrows do *and* re-styles the mouse
to match, so picking PAN makes dragging pan too rather than leaving the two
gestures inconsistent.

Six named views: **iso** (the default three-quarter), **flow** (side-on across
the 590 mm axis — the view that shows the row-to-row gradient), **inlet** and
**exhaust** (head-on at the blue and green cutouts), **top**, and **side**.

Pitch is clamped to ±85° so the camera never passes through a pole, which would
flip the view-up and invert the controls. Every rotation re-orthogonalises the
view-up, so repeated pitch steps cannot slowly roll the horizon.

Three trame details worth recording, since each silently produced a dead control:

* The keydown handler is a **raw JS string**. Passing the `(expr,)` tuple form
  makes trame wrap it in `trigger(...)`, firing a trigger *named by the
  expression* instead of assigning to `cam_key` — no key ever reaches the server.
* `preventDefault()` is called **inside the handler**, not via a `.prevent`
  modifier: trame renders `keydown_prevent` as `@keydown-prevent`, which Vue
  does not read as a modifier, so the arrows would still scroll the page.
* The key value carries a **`#timestamp` suffix**, because trame skips a state
  change when the value is unchanged — without it, pressing the same arrow twice
  in a row would register only once.

## Appearance

Panels, electronics shelf, PCBs and internal structure start **translucent** so
the cells read; one switch makes them solid. **Depth peeling** is enabled, so
transparency is correct at any opacity — `DRAW_ORDER` alone cannot work once
opacity changes at runtime, because the correct order depends on the values and
the actors are already added.

The colour scale is **pinned** (default 40–50 °C) rather than auto-scaled per
frame, so the three modes are visually comparable; without that, a cell holding
45 °C changes colour as its neighbours warm. The 35–65 °C full span was rejected
as a default because it renders a realistic result near-black.

---

## Extra heat sources

At 61.2 A: contactor $I^2R$ 1.5 W, coil hold 16 W, PCBs 15 W → **≈32.5 W, 11% of
the 296.9 W cell load** — all **ASSUMED**, none in the design review.

Measured placement decides how they enter: the 4 Gigavacs sit at z = −583…−526
(at the green plate) and both boards at y = 196…198 (on the shelf), so **both are
outside the cell air path**. They raise the *exhaust* by ~0.65 K and leave peak
cell temperature untouched. Adding them to the cell load would have inflated every
cell temperature by ~11%.

---

## What changed in the existing code

| file | change |
|---|---|
| `geometry.py` | `load_or_build(path=...)`; cache key includes the file stem; name-based classification with new `pcb`/`shelf` classes; `named_parts()` |
| `params.py` | `CellThermal`, `ElectronicsHeat`, 13 new provenance rows; **removed dead code** after `__main__` that shadowed the fan curves with a different 13-point list |
| `plots.py` | `sensitivity_figure` now honours the caller's geometry (it silently rebuilt defaults before) |

**Classification bugs fixed:** `Shelf Panel` (274 × 8.2 × 454 mm) and the
`MainBMS PCB` (158 × 1.6 × 249 mm) both tripped the "big and thin" plate
heuristic and were being classed as *enclosure*, which would have made them
impossible to toggle separately. Busbar keywords now test before the divider
rule. `Front Duct` and `Top Brace` previously fell through to `other`.

## Verification

| check | result |
|---|---|
| network module reproduces `flow.py` | **exact, 0.0e+00 CFM difference** |
| legacy `BoxAssembly.step` classification | unchanged (7 panels, 408 cells) |
| new assembly | 408 cells; all 10 `case_plate` are genuine enclosure parts |
| transient → analytic steady state | max error **2.8e-12 K** |
| exponential approach at $t=\tau$ | 0.6336 vs 1−1/e = 0.6321 (0.15%) |
| transient energy balance | **1.1e-15** |
| both-panels-in | raises `InfeasibleTopology` |
| all 17 equations render offline | 17/17 |
| `out/summary.png` | byte-identical to before |

## Limitations worth knowing

- Bank Δp is ~98% of the total and rests on extrapolating the Žukauskas friction
  curve from $P_T = 1.25$ to the actual 1.114. **This single extrapolation
  dominates every number in all three modes.**
- `contact_conductance` (0.02 W/K) is the weakest assumed value and the first
  thing worth measuring.
- No CFD or experimental validation exists for the push–pull case; series
  composition assumes the two fan sets do not aerodynamically interfere
  (reasonable at 590 mm apart with the bank between them, but unverified).
- Which direction the fans actually blow is still not settled by the CAD.

---

## A bug worth recording: the blank control drawer

The first build shipped with the entire left control panel rendering **empty** —
every slider, switch and toggle missing — while the server started normally, the
3D view drew and the figure sidebar updated. The mode buttons were also drawn
twice.

Two separate trame traps, both of which **fail silently**:

1. **`VExpansionPanels(model_value=[0, 1, 2])`.** trame calls `.startswith()` on
   each entry, raises on the integers, catches the exception internally, and
   emits `<VExpansionPanels html-error />` — discarding the entire subtree. The
   only symptom is five `'int' object has no attribute 'startswith'` lines on
   startup, which look like library noise. Use a list of string panel `value`s
   bound through `v_model` instead.

2. **`Container(children=[...])`.** Constructing a trame widget already registers
   it with the open parent, so passing the same instances via `children` renders
   each one **twice**. Use the context-manager form.

`src/test_ui.py` now guards both:

```bash
.venv/bin/python src/test_ui.py
```

It builds the real UI, asserts `html.count("html-error") == 0`, checks all 26
control bindings and 9 markup elements are present, and verifies the mode toggle
has exactly 3 buttons rather than 6. Re-introducing either bug makes it fail.

### And a third: Play did nothing while Step worked

`animate()` ended its loop body with `await asynchronous.sleep(0.06)`, but
**there is no `sleep` in `trame.app.asynchronous`**. The coroutine raised
`AttributeError` on its first iteration, inside a background task where nothing
surfaces the exception — so pressing Play advanced exactly one invisible step and
then stopped, while Step (synchronous, no `await`) kept working perfectly. That
asymmetry is the diagnostic fingerprint.

Fixed by using `asyncio.sleep`, moving the render inside the `with state:` block
so the flushed state and the new frame arrive together, and wrapping the loop in
`try/except/finally` that logs any exception and always clears `state.playing`
so the button cannot stick.

`src/test_ui.py` now drives `animate()` for real and asserts simulated time
advances, so a dead Play button fails the test rather than reaching you.

**The general lesson:** "the server started and nothing crashed" is a much weaker
check than it feels like when the deliverable is a UI. Verify the widgets are
actually in the rendered output and that the interactions actually do something;
treat any warning during UI construction as a failure to be explained rather than
noise to be dismissed. All three of these bugs were invisible from the server
side and obvious within seconds of a human using the app.
