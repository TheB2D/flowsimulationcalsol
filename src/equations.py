r"""
The governing equations, rendered for display in the app.

Rendered with matplotlib's built-in mathtext rather than MathJax or KaTeX, for
three reasons: the app runs locally and a CDN `<script>` would leave the panel
blank with no network; matplotlib is already a dependency, so nothing new is
vendored; and going through the same rcParams as `plots.py` makes the maths
typographically identical to the figures beside it.

mathtext is not full LaTeX -- no `\text{}`, no `align` environments -- so each
entry is a single inline expression using `\mathrm{}`.  The test at the bottom
renders every entry, because a parse error would otherwise only surface when a
user opened the panel.
"""

from __future__ import annotations

import base64
import functools
import io
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

INK = "#18181B"
MUTED = "#64748B"


# (key, group, caption, latex, source)
EQUATIONS: list[tuple[str, str, str, str, str]] = [
    ("operating_point", "Flow", "Fan operating point",
     r"\Delta p_{\mathrm{fan}}(Q) = \Delta p_{\mathrm{sys}}(Q)",
     "the intersection the review left open"),

    ("series", "Flow", "Fans in series (push-pull): pressures add",
     r"\sum_i \Delta p_i(Q) = \Delta p_{\mathrm{sys}}(Q)",
     "derived -- the basis of the two-sided mode"),

    ("parallel", "Flow", "Fans in parallel on one face: flows add",
     r"\sum_i Q_i(\Delta p) = Q",
     "derived"),

    ("continuity", "Flow", "Mass balance over the box",
     r"\sum_{\mathrm{in}} Q_i = \sum_{\mathrm{out}} Q_j",
     "no solution if every fan blows inward"),

    ("bank_dp", "Flow", "Bank pressure drop (Zukauskas)",
     r"\Delta p_{\mathrm{bank}} = N_L\,\chi\,f\,\frac{\rho V_{\max}^2}{2}",
     "Incropera 7e, Fig. 7.14"),

    ("vmax", "Flow", "Maximum velocity in the minimum section",
     r"V_{\max} = V_{\infty}\,\frac{S_T}{S_T - D}",
     "continuity through A_min"),

    ("nusselt", "Heat", "Nusselt number for a staggered bank",
     r"\mathrm{Nu} = C_1 C_2\,\mathrm{Re}_{\max}^{m}\,\mathrm{Pr}^{0.36}"
     r"\left(\frac{\mathrm{Pr}}{\mathrm{Pr}_s}\right)^{1/4}",
     "Incropera 7e, Table 7.7"),

    ("h_conv", "Heat", "Convective coefficient",
     r"h = \frac{\mathrm{Nu}\,k}{D}",
     "derived"),

    ("air_march", "Heat", "Air energy balance, row by row",
     r"\dot m\,c_p\,\Delta T_{\mathrm{air}} = \sum \dot Q_{\mathrm{cell}}",
     "steady march along the flow"),

    ("cell_T", "Heat", "Cell surface temperature (steady)",
     r"T_{\mathrm{cell}} = T_{\mathrm{air}} + "
     r"\frac{\dot Q_{\mathrm{cell}}}{h\,A_{\mathrm{cell}}}",
     "derived"),

    ("transient", "Transient", "Lumped-node energy equation",
     r"C_i\,\frac{dT_i}{dt} = \dot q_i - h_i A_i (T_i - T_{\mathrm{air}})"
     r" + \sum_j G_{ij}\,(T_j - T_i)",
     "the new physics in mode 3"),

    ("scatter", "Transient", "Manufacturing scatter (physical)",
     r"\dot q_i = I_{\mathrm{cell}}^2 R_i,\quad "
     r"R_i = R_{\mathrm{int}}(1 + \epsilon_i),\quad "
     r"\epsilon_i \sim \mathcal{N}(0,\sigma_R^2)",
     "seeded, reproducible"),

    ("ou_noise", "Transient", "Additive noise (visual only, off by default)",
     r"d\xi_i = -\frac{\xi_i}{\tau}\,dt + \sqrt{\frac{2}{\tau}}\,dW_i",
     "Ornstein-Uhlenbeck -- NOT physics"),

    ("stability", "Transient", "Explicit-Euler stability limit",
     r"\Delta t < \min_i \frac{C_i}{h_i A_i + \sum_j G_{ij}}",
     "checked at runtime"),

    ("biot", "Transient", "Biot number -- lumped validity",
     r"\mathrm{Bi} = \frac{h\,(D/4)}{k_{\mathrm{radial}}}",
     "Bi > 0.1 means the node is a volume mean"),

    ("radial", "Transient", "Centre-to-surface correction",
     r"\Delta T_{\mathrm{radial}} = \frac{\dot q}{4\pi k L}",
     "added before comparing to the 60 C surface limit"),

    ("energy", "Checks", "Transient energy balance",
     r"\sum_i C_i\,\Delta T_i = \int\left(\sum_i \dot q_i - "
     r"\dot m c_p \Delta T_{\mathrm{air}}\right)dt",
     "conduction cancels because G is symmetric"),
]

GROUPS = ["Flow", "Heat", "Transient", "Checks"]


@functools.lru_cache(maxsize=128)
def equation_png(latex: str, fontsize: float = 15.0, dpi: int = 180,
                 color: str = INK) -> bytes:
    """Render one expression to transparent PNG bytes."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def equation_uri(latex: str, **kw) -> str:
    """Render to a data: URI an <img> can show directly."""
    return ("data:image/png;base64,"
            + base64.b64encode(equation_png(latex, **kw)).decode())


def catalogue(group: str | None = None) -> list[dict]:
    """Every equation, ready for the UI, optionally filtered by group."""
    out = []
    for key, grp, caption, latex, source in EQUATIONS:
        if group and grp != group:
            continue
        out.append({"key": key, "group": grp, "caption": caption,
                    "latex": latex, "source": source,
                    "uri": equation_uri(latex)})
    return out


def live_values(model=None, result=None, transient=None) -> list[tuple[str, str]]:
    """Current numeric values beside the symbols.

    Showing the live numbers next to the symbolic form is what makes the panel
    a reading of this model rather than a decorative wall of maths.
    """
    from params import KELVIN
    rows: list[tuple[str, str]] = []
    if result is not None:
        rows += [
            ("Q", f"{result.flow_cfm:.1f} CFM"),
            ("V_max", f"{result.v_max:.2f} m/s"),
            ("Re_max", f"{result.reynolds_max:.0f}"),
            ("Nu", f"{result.nusselt:.1f}"),
            ("h", f"{result.h_conv:.1f} W/m^2.K"),
            ("dp_bank", f"{result.pressure_drop_bank:.0f} Pa"),
            ("dp_total", f"{result.pressure_drop_total:.0f} Pa"),
            ("T_cell,max", f"{result.cell_surface_temperature - KELVIN:.2f} C"),
        ]
    if transient is not None:
        rows += [
            ("C = m c_p", f"{transient.C:.1f} J/K"),
            ("tau = C/hA", f"{transient.tau:.1f} s"),
            ("dt_max", f"{transient.dt_max:.1f} s"),
            ("Bi", f"{transient.biot:.2f}"
                   + ("  (>0.1: volume mean)" if transient.biot > 0.1 else "")),
        ]
    return rows


if __name__ == "__main__":
    ok = 0
    for key, grp, caption, latex, source in EQUATIONS:
        try:
            png = equation_png(latex)
            ok += 1
            print(f"  [ok] {grp:9s} {key:16s} {len(png):6d} B  {caption}")
        except Exception as exc:
            print(f"  [FAIL] {key}: {exc}")
    print(f"\n{ok}/{len(EQUATIONS)} equations render")
