# Airflow either way: "from the 17" vs "from the 12"

Which face the cooling air enters is **not settled by the CAD** — the fans and
ducts were suppressed before the STEP export, so nothing in `BoxAssembly.step`
marks an inlet. Rather than pick one, `src/direction.py` builds both and lets you
toggle.

```bash
.venv/bin/python src/direction.py                  # both, side by side
.venv/bin/python src/direction.py --only from_17   # one, in detail
.venv/bin/python src/direction.py --verify         # re-derive both from the CAD
.venv/bin/python src/direction.py --export         # out/flow_from_{12,17}.json

# colour the 3D render by either field
.venv/bin/python src/render.py --temperature --temps "$PWD/out/flow_from_17.json"
```

`--temps` needs an **absolute path**: `render.py` silently falls back to a
placeholder gradient if the file is not found, and prints
`no temperature file given -- showing a PLACEHOLDER gradient` when it does.

## The two directions

The 408 cells are one staggered lattice. A staggered bank is counted by its
**offset planes** — the successive tube planes the air crosses, each offset half
a transverse pitch from the last. Applying that consistently to the same lattice,
viewed two ways:

| | **from the 12** (flow ∥ z) | **from the 17** (flow ∥ x) |
|---|---|---|
| air enters | the 12-wide face | the 17-wide face |
| $S_T$ across the flow | 20.50 mm | **50.40 mm** |
| $S_L$ plane to plane | 25.20 mm | 10.25 mm |
| $N_L$ planes crossed | 17 | 24 |
| $N_T$ per plane, per bank | 12 | 8.5 (9/8 alternating) |
| flow front | 246.0 mm | 428.4 mm |
| minimum gap | **2.10 mm** (transverse) | **17.61 mm** (diagonal) |
| $A_{\min}/A_{\text{face}}$ | 0.1024 | 0.3494 |

The "from the 17" row is the one that is easy to get wrong. Viewed along $x$ the
lattice does **not** present 12 planes of 34 cells at the 25.2 mm row pitch. It
resolves into **24 planes at 10.25 mm**, each holding 8 or 9 cells at **50.4 mm**
pitch. So the transverse pitch that direction is 50.4 mm, not 25.2 mm — and the
minimum-area plane switches from transverse to diagonal.

Both descriptions recover the invariants they must, which is the check that they
are the same lattice: **408 cells**, **1.5330 m² of wetted area**, and
$S_D = 27.20$ mm. `--verify` re-derives all of this from `out/cell_positions.csv`
and asserts it; all nine checks pass.

## Results (worst case: 40 °C inlet, ~297 W, 3× Delta GFB0812ES-E)

| | from the 12 | from the 17 | |
|---|---|---|---|
| flow | 93.2 CFM | **359.7 CFM** | 3.86× |
| face velocity | 1.38 m/s | 3.05 m/s | |
| $V_{\max}$ | 13.43 m/s | 8.72 m/s | 0.65× |
| $Re_{\max}$ | 14 373 | 9 447 | |
| $\Delta p$ bank | 1422 Pa | 234 Pa | 0.16× |
| $\Delta p$ total | 1455 Pa | 729 Pa | 0.50× |
| $h$ | 136.7 W·m⁻²K⁻¹ | 126.9 W·m⁻²K⁻¹ | 0.93× |
| air rise | 5.95 °C | 1.54 °C | |
| **hottest cell** | **47.0 °C** | **43.0 °C** | **−4.0 °C** |
| margin to 45 °C goal | −2.0 °C (missed) | +2.0 °C (met) | |
| margin to 60 °C limit | +13.0 °C | +17.0 °C | |

**Both pass the 60 °C limit; only "from the 17" meets the 45 °C design goal.**

### Why the difference is so large

$\Delta p_{\text{bank}} = N_L \chi f \tfrac{1}{2}\rho V_{\max}^2$, and $V_{\max}$
is set by the minimum free area. Turning the flow 90° widens the transverse pitch
from 20.5 mm to 50.4 mm, which opens $A_{\min}/A_{\text{face}}$ by 3.4× and
widens the flow front by 1.74×. Against a nearly-flat fan curve the bank resists
6× less and passes 3.9× the air.

Note the trade: $h$ actually *drops* slightly (127 vs 137 W·m⁻²K⁻¹) because
$V_{\max}$ is lower. The cooling win is **not** better heat transfer per cell — it
is the 3.9× mass flow, which cuts the air temperature rise from 5.95 °C to
1.54 °C. The hot end of the pack is what improves.

Also worth flagging: in "from the 17" the duct and grille losses (~495 Pa) now
**exceed** the bank (234 Pa), because the same fixed fan area passes 3.9× the
flow and those losses go as $V_{\text{fan}}^2$. The `K = 3.5 + 1.4` assumption was
inherited from the prior model, where it was negligible; in this direction it is
half the total, so it is worth revisiting before trusting the ~360 CFM figure.

## What this does and does not change

`src/direction.py` **reuses** `src/flow.py` — it does not reimplement the physics.
`PackGeometry` is already fully parameterised and the Žukauskas correlations
already take $S_T$ and $S_L$, so each direction is just a different geometry
handed to the same solver. Running the model in the `from_12` direction
reproduces `flow.py`'s standalone result to 3 parts in $10^9$ (Brent tolerance).

One thing that could not be reused: `PackFlowModel.cell_temperatures` indexes on
the CSV's `row` column, which is the z-row — correct only for `from_12`. The
direction module bins cells along whichever coordinate the air actually follows.

`n_parallel` in the geometry is the **geometric** count across the flow front, not
the electrical 12P. The two coincide only in the `from_12` direction; the
electrical configuration lives in `PackElectrical` and is untouched.

## Caveats

- **Which direction is real is still an open question.** Confirm against the
  design review before relying on either. This tool answers "what if", not "what is".
- The friction factor is extrapolated from Incropera's tightest published
  staggered curve ($P_T = 1.25$); at $P_T = 1.114$ the `from_12` direction is
  outside it, which the prior model already flags as its largest uncertainty. The
  `from_17` direction at $P_T = 2.74$ is extrapolated the other way, so its
  $\Delta p$ is also uncertain — though the bank matters much less there.
- The duct/grille $K$ values are assumptions, and they dominate `from_17` (above).
- 1-D row march: uniform flow across the face, no bypass around the bank, no
  recirculation, heat spread evenly over all cells.
- Real packs have edge bypass, which would hurt `from_17` more than `from_12`
  because its bank resistance is lower relative to any leak path.

## A note on reproducibility

`src/direction.py` imports `params.py` and `flow.py` rather than copying them, so
its numbers track any change to the shared parameters. The figures above were
taken at one point in time; re-run `--export` after any parameter edit rather than
trusting the table.
