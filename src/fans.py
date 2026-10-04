r"""
Fans drawn in 3D, at the openings they would actually be mounted in.

Positions come from the measured cutouts in `openings.py`, so a fan is never
drawn somewhere a fan could not go.  An 80 mm frame fits three times across the
251 mm blue cutout and three times across the 244 mm green one, which is the
`max_fans` limit the placement search respects.

The glyphs are cheap (a few thousand triangles each) and are rebuilt whenever
the arrangement changes.  Spin is applied by rotating the ACTOR, not by
rebuilding geometry, so an animated fan costs nothing per frame.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from openings import Opening, FRONT_PLATE, REAR_PLATE

FRAME_M = 0.080
HUB_FRAC = 0.40
BLADES = 7

COLOR_IN = "#38BDF8"      # intake  -- cool blue
COLOR_OUT = "#FB7185"     # exhaust -- warm red
COLOR_FRAME = "#CBD5E1"


def fan_centres(opening: Opening, n: int, centre_xy=None) -> list:
    """Evenly spaced fan centres across an opening, in CAD mm.

    The default is the opening's own MEASURED cutout centre.  Do not pass the
    panel's centroid: the cutout sits low on both plates, so the centroid is
    75-85 mm higher than the hole and fans placed there float above it.
    """
    if n <= 0:
        return []
    w = opening.width * 1e3
    cx, cy, cz = opening.centre_mm
    if centre_xy is not None:
        cx, cy = centre_xy
    pitch = w / n
    x0 = cx - w / 2 + pitch / 2
    return [(x0 + k * pitch, cy, cz) for k in range(n)]


def fan_glyph(centre, normal=(0, 0, 1), frame_m: float = FRAME_M,
              blades: int = BLADES, sense: int = +1) -> pv.PolyData:
    """One fan: frame ring, hub, and swept blades, oriented to `normal`.

    Built in local coordinates with the axis along +z, then transformed once.
    """
    r = frame_m * 1e3 / 2.0
    rh = r * HUB_FRAC

    parts = [
        pv.Disc(center=(0, 0, 0), inner=r * 0.88, outer=r,
                normal=(0, 0, 1), c_res=40),
        pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                    radius=rh, height=r * 0.30, resolution=24),
    ]
    for k in range(blades):
        a = 2 * math.pi * k / blades
        blade = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                         i_size=(r - rh) * 0.92, j_size=r * 0.34,
                         i_resolution=1, j_resolution=1)
        blade.rotate_z(math.degrees(a) + 18.0, inplace=True)
        blade.translate(((rh + (r - rh) / 2) * math.cos(a),
                         (rh + (r - rh) / 2) * math.sin(a), 0), inplace=True)
        parts.append(blade)

    glyph = parts[0].merge(parts[1:])

    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    z = np.array([0.0, 0.0, 1.0])
    if not np.allclose(n, z):
        axis = np.cross(z, n)
        if np.linalg.norm(axis) < 1e-9:
            glyph.rotate_x(180.0, inplace=True)
        else:
            ang = math.degrees(math.acos(float(np.clip(np.dot(z, n), -1, 1))))
            glyph.rotate_vector(vector=axis / np.linalg.norm(axis),
                                angle=ang, inplace=True)
    glyph.translate(np.asarray(centre, dtype=float), inplace=True)
    return glyph


def flow_arrow(centre, normal=(0, 0, 1), sense: int = +1,
               length_mm: float = 110.0) -> pv.PolyData:
    """An arrow showing which way this face moves air.

    Points INTO the box for an inlet and out of it for an outlet, which is the
    quickest way to read a push-pull arrangement at a glance.
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    d = n * (-1.0 if sense > 0 else 1.0)      # inlet pushes inward
    start = np.asarray(centre, dtype=float) - d * length_mm * 0.5
    return pv.Arrow(start=start, direction=d, scale=length_mm,
                    tip_length=0.3, tip_radius=0.11, shaft_radius=0.035)


def build_fan_actors(plotter, topology, panel_centres: dict | None = None,
                     show_arrows: bool = True) -> list:
    """Draw every fan in a topology. Returns the actors so they can be removed.

    Positions come from each opening's measured cutout centre.  `panel_centres`
    is accepted only as an override and is normally left as None -- an earlier
    version fed it the panel CENTROID, which sits 75-85 mm above the hole and
    left every fan floating clear of the cutout.
    """
    actors = []
    for fs in topology.sets:
        if not fs.active:
            continue
        key = "front" if fs.opening is FRONT_PLATE else "rear"
        cxy = (panel_centres or {}).get(key)
        normal = (0, 0, 1.0 * fs.opening.normal_sign)
        colour = COLOR_IN if fs.sense > 0 else COLOR_OUT

        for c in fan_centres(fs.opening, fs.n, cxy):
            actors.append(plotter.add_mesh(
                fan_glyph(c, normal, sense=fs.sense),
                color=colour, opacity=0.95, smooth_shading=True))
            if show_arrows:
                actors.append(plotter.add_mesh(
                    flow_arrow(c, normal, fs.sense),
                    color=colour, opacity=0.55))
    return actors


def panel_centres_from_bodies(bodies) -> dict:
    """Locate the two cut panels in the CAD so fans land on them."""
    import geometry as G
    out = {}
    for b in bodies:
        key = G.normalise_part_name(b.name)
        if key == "front plate":
            out["front"] = (b.cx, b.cy)
        elif key == "rear plate":
            out["rear"] = (b.cx, b.cy)
    return out
