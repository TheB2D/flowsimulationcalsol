"""
The real openings in the battery box, measured from the CAD.

`Battery Box.step` is the only one of the four exported STEP files with true
B-rep solids rather than tessellated meshes, so it is the only one in which a
cutout is a real inner wire on a planar face and its area can be measured
rather than estimated.  Everything in this module came out of that file.

The two openings the user asked about:

    Front Plate-1   the BLUE panel,  cutout 251.3 x 121.0 mm
    Rear Plate-1    the GREEN panel, cutout 244.0 x 124.0 mm

They sit on opposite ends of the z axis, 590 mm apart, with the cell block
between them.  Each is 330 x 295 x 7.15 mm.

Two things worth knowing before using these numbers:

  * The green cutout is partly blocked by the purple `Rear Panel-1` truss, so
    its POROSITY is 0.886, not 1.0.  The blue one is clear.
  * The blue cutout (251.3 x 121.0) is a 2% width match to the cell bank face
    (246.0 x 130.0).  The openings were sized to the bank, which is what fixes
    the flow axis as z -- i.e. `Direction.FROM_12`.

Identify panels by NAME, never by colour: OCCT's XCAF returns *linear* RGB, so
the blue panel reads back as #000BA4 rather than the #023DD2 in the STEP.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@dataclass(frozen=True)
class Opening:
    """One rectangular opening in the enclosure.

    Lengths in metres.  `plane_z` is the panel's position on the flow axis;
    `normal_sign` is +1 if the outward normal points along +z.

    `centre_x` / `centre_y` are the centre of the CUTOUT, which is NOT the
    centre of the panel: the cutout sits low on both plates, and the panel's
    own centroid is dragged ~75-85 mm higher by the material above it.  Placing
    anything on the panel centroid puts it well clear of the hole.
    """

    name: str
    panel: str                 # the CAD part name it is cut into
    plane_z: float             # m
    width: float               # m, across x
    height: float              # m, across y
    centre_x: float = 0.0      # m, centre of the cutout
    centre_y: float = 0.0      # m
    porosity: float = 1.0      # open fraction after any internal obstruction
    max_fans: int = 3          # 80 mm frames that fit across the width
    normal_sign: int = +1
    source: str = "MEASURED from Battery Box.step"

    @property
    def centre_mm(self) -> tuple[float, float, float]:
        """Cutout centre in CAD millimetres -- where fans actually belong."""
        return (self.centre_x * 1e3, self.centre_y * 1e3, self.plane_z * 1e3)

    @property
    def gross_area(self) -> float:
        """Cutout area ignoring obstruction [m^2]."""
        return self.width * self.height

    @property
    def free_area(self) -> float:
        """Area actually open to flow [m^2]."""
        return self.gross_area * self.porosity

    def entry_loss_coefficient(self, bank_face_area: float,
                               k_grille: float = 1.4) -> float:
        r"""Loss coefficient for air passing this opening, referenced to the
        velocity in the OPENING.

        Two contributions:

        * a sudden area change between the opening and the bank face,
          $K \approx (1 - A_1/A_2)^2$ for an expansion;
        * a grille/obstruction term scaled as $1/\sigma^2$, since blocking the
          face to a fraction $\sigma$ raises the local velocity by $1/\sigma$
          and the dynamic head by its square.

        This is a refinement, not a rewrite: duct and grille losses are only
        ~2% of the total drop, which the cell bank dominates at ~98%.  The
        default K-factors in `OperatingConditions` remain the fallback.
        """
        a_ratio = self.free_area / bank_face_area if bank_face_area > 0 else 1.0
        k_area = (1.0 - min(a_ratio, 1.0 / max(a_ratio, 1e-9))) ** 2
        k_block = k_grille / max(self.porosity, 1e-3) ** 2
        return k_area + k_block

    def fan_capacity(self, frame_m: float = 0.080) -> int:
        """How many fan frames fit across the opening's width."""
        return max(1, int(self.width // frame_m))

    def describe(self) -> str:
        return (f"{self.name:12s} {self.width*1e3:6.1f} x {self.height*1e3:6.1f} mm"
                f"  centre ({self.centre_x*1e3:+.0f}, {self.centre_y*1e3:+.0f})"
                f"  gross {self.gross_area*1e4:6.1f} cm^2"
                f"  porosity {self.porosity:.3f}"
                f"  free {self.free_area*1e4:6.1f} cm^2"
                f"  fits {self.max_fans} fans")


# -- the measured openings ----------------------------------------------------

# Cutout extents measured from the inner wires of the planar faces:
#   front  x -125.6..125.6, y  17.9..138.9   -> centre (0.0,  78.4)
#   rear   x -122.0..122.0, y  11.1..135.2   -> centre (0.0,  73.1)
# Both centres sit well below their panel's centroid (152.7 / 156.9 mm).
FRONT_PLATE = Opening(
    name="blue (front)", panel="Front Plate-1",
    plane_z=0.000, width=0.2513, height=0.1210,
    centre_x=0.0000, centre_y=0.0784,
    porosity=1.000, max_fans=3, normal_sign=+1)

REAR_PLATE = Opening(
    name="green (rear)", panel="Rear Plate-1",
    plane_z=-0.590, width=0.2440, height=0.1240,
    centre_x=0.0000, centre_y=0.0731,
    porosity=0.886, max_fans=3, normal_sign=-1)

FRONT_DUCT = Opening(
    name="front duct", panel="Front Duct-1",
    plane_z=-0.032, width=0.2473, height=0.1170,
    centre_x=0.0000, centre_y=0.0784,
    porosity=1.000, max_fans=3, normal_sign=+1)

# The two faces a fan can actually be mounted on.  FRONT_DUCT sits 32 mm behind
# FRONT_PLATE and is in SERIES with it, so it is not an independent site.
OPENINGS = {"front": FRONT_PLATE, "rear": REAR_PLATE, "duct": FRONT_DUCT}
FAN_FACES = ("front", "rear")


def summary() -> str:
    lines = ["Measured openings (Battery Box.step):"]
    for key in ("front", "rear", "duct"):
        lines.append("  " + OPENINGS[key].describe())
    sep = abs(FRONT_PLATE.plane_z - REAR_PLATE.plane_z)
    lines.append(f"  front and rear are {sep*1e3:.0f} mm apart on the z axis")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
