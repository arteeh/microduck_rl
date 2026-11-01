"""A simulated VL53L5CX / VL53L8CX, so the duck can see a wall.

8x8 zones over a 45-degree square field of view, out to 4 m, at 15 Hz — the real sensor's shape,
because `tofd` publishes frames of exactly that and `maploc` reprojects them assuming it.

**A zone sees a cone, not a ray.** Each zone integrates the light returned from its whole ~5.6°
footprint into a histogram and reports the peaks it finds there, so this casts a grid of rays per
zone and builds the zone's targets from what they hit, the way the sensor's histogram processing
does (ST UM2884 §4.9–4.10, UM3109 §4.9–4.10):

- up to four targets per zone (`TARGETS`), strongest first — the ULD's default target order;
- two surfaces less than 600 mm apart along the zone are one target ("The minimum distance between
  two targets to be detected is 600 mm", UM3109 rev 7 §4.10, UM2884 from rev 3), reported at their
  signal-weighted distance with status 9, "range valid with large pulse (may be due to a merged
  target)" (UM2884 table 4) — the edge of a table half a metre in front of a wall is one smeared
  range, not two;
- farther apart, both are reported: the table edge *and* the wall behind it.

Signal per ray is a Lambertian return, cos(incidence) / range², every surface the same reflectance
(MuJoCo materials carry no infrared reflectance). A target is detected when its signal is at least
what a surface filling the whole zone face-on returns from the far limit (4 m by default). That floor is an estimate,
not a datasheet number: the real limit depends on reflectance and ambient light. Not modelled:
veiling glare into neighbouring zones (what ST's sharpener trims), cover-glass crosstalk, the
spurious multipath targets ST warns of indoors, and statuses other than 5, 9 and 255.

Range noise grows with distance as `~/MISC/microduck_maploc`'s `sim/tof_sensor.py` had it from the
datasheet and a sensor on a desk: millimetres up close, centimetres out near the limit.

**The status byte matters as much as the distance.** A real sensor distinguishes "nothing out there"
from "could not measure", and `maploc` treats them differently — a zone with no target is empty
space to clear on the map, and a zone that failed is no information at all.
"""

from __future__ import annotations

from itertools import pairwise

import mujoco
import numpy as np

# The sensor, as `tof/src/lib.rs` publishes it: ROWS x COLS zones of TARGETS slots each.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
TARGETS = 4
STATUS_VALID = 5
STATUS_MERGED = 9
STATUS_NO_TARGET = 255

# VL53L5CX and VL53L8CX: 45 degrees per axis (63 and 65 on the diagonal), 4 m of range.
FOV_DEG = 45.0
MAX_RANGE = 4.0
# ST: the minimum distance between two targets to be detected in one zone.
MIN_SEPARATION = 0.6
# Rays per zone along each axis.
SAMPLES = 5
# Returns this far apart along a zone are different surfaces, closer ones one surface (a wall or a
# floor seen at an angle spans a range across the footprint). An estimate.
SURFACE_GAP = 0.15


class Tof:
    """The 8x8 sensor on one duck's `tof` site.

    Rays are cast in the site's frame — +x forward, +y left, +z up — so a head that turns takes the
    sensor with it, which is the whole point of `robot.look` scanning a room.
    """

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0, noise: float = 1.0,
                 minimum: float = 0.0, maximum: float = MAX_RANGE):
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("ToF noise ratio must be finite and nonnegative")
        self.minimum, self.maximum = minimum, maximum
        self.noise = noise
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)
        # Signal of a surface filling a zone face-on at the far limit: the faintest target seen.
        self.floor = 1.0 / maximum**2

        # SAMPLES² rays spread evenly over each zone's footprint, zone-major. Azimuth about +z
        # (positive toward +y, the duck's left) and elevation up — row 0 is the top of the frame,
        # and column 0 the sensor's LEFT, as `kinematics::tof` reads the real sensor's buffer.
        # With column 0 on the right every frame reached the mapper mirrored.
        half = np.radians(FOV_DEG) / 2.0
        step = 2.0 * half / COLS
        within = ((np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5) * step
        centres = -half + (np.arange(COLS) + 0.5) * step
        directions = np.zeros((ZONES, SAMPLES * SAMPLES, 3))
        for row in range(ROWS):
            for col in range(COLS):
                elevation = -(centres[row] + within)[:, None]
                azimuth = -(centres[col] + within)[None, :]
                ray = np.stack(np.broadcast_arrays(
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ), axis=-1)
                directions[row * COLS + col] = ray.reshape(-1, 3)
        self.directions = directions.reshape(-1, 3)
        count = len(self.directions)
        self._geom = np.zeros(count, dtype=np.int32)
        self._dist = np.zeros(count)
        self._normal = np.zeros(3 * count)
        self._still = mujoco.MjData(model)

    def capture(self, data: mujoco.MjData) -> None:
        """Take the world as it is now. The only part that reads `data`, and a copy rather than the
        ray casts themselves, so the caller holds the world still for microseconds, not the
        milliseconds `frame` casts for."""
        mujoco.mj_copyData(self._still, self.model, data)

    def _cast(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Every ray of the captured world: range in metres (-1 for no hit) and the cosine of its
        incidence. Self-hits are kept: a real sensor sees the duck's own beak when the beak is in
        front of it."""
        data = self._still
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = self.directions @ rotation.T
        # A zero-length direction makes `mj_ray` abort the whole process — "vector length is too
        # small" — so it is checked rather than trusted: before the first forward pass every site
        # orientation is zero, and a client can connect before then.
        if not np.isfinite(world).all() or np.linalg.norm(world[0]) < 1e-9:
            return None
        mujoco.mj_multiRay(
            self.model, data, origin, np.ascontiguousarray(world).reshape(-1), None, 1, -1,
            self._geom, self._dist, self._normal, len(world), self.maximum,
        )
        normal = self._normal.reshape(-1, 3)
        return self._dist, np.abs(np.einsum("ij,ij->i", normal, world))

    def frame(self) -> tuple[list[int], list[int]]:
        """The captured world's targets: distances in millimetres and statuses, ZONES x TARGETS
        slots zone-major, strongest target first; empty slots are status 255."""
        rays = self._cast()
        distance_mm = [0] * (ZONES * TARGETS)
        status = [STATUS_NO_TARGET] * (ZONES * TARGETS)
        if rays is None:
            return distance_mm, status
        hit, incidence = (a.reshape(ZONES, -1) for a in rays)
        per_ray = 1.0 / (SAMPLES * SAMPLES)
        for zone in range(ZONES):
            seen = (hit[zone] >= max(self.minimum, 1e-3)) & (hit[zone] <= self.maximum)
            if not seen.any():
                continue
            r = hit[zone][seen]
            order = np.argsort(r)
            r = r[order]
            w = (incidence[zone][seen][order] / (r * r)) * per_ray
            # Surfaces: runs of returns with no gap wider than SURFACE_GAP between them.
            cuts = np.flatnonzero(np.diff(r) > SURFACE_GAP) + 1
            # Each as [signal, signal-weighted range sum, surfaces merged into it].
            peaks = [[float(sw.sum()), float((sr * sw).sum()), 1]
                     for sr, sw in zip(np.split(r, cuts), np.split(w, cuts), strict=True)]
            # Peaks closer than ST's minimum separation are one target.
            while len(peaks) > 1:
                gaps = [b[1] / b[0] - a[1] / a[0] for a, b in pairwise(peaks)]
                i = int(np.argmin(gaps))
                if gaps[i] >= MIN_SEPARATION:
                    break
                a, b = peaks[i], peaks.pop(i + 1)
                peaks[i] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            peaks = sorted((p for p in peaks if p[0] >= self.floor), key=lambda p: -p[0])
            for slot, (signal, weighted, merged) in enumerate(peaks[:TARGETS]):
                at = weighted / signal
                # Noise that grows with range, as the datasheet has it: a few millimetres up close,
                # a couple of centimetres at the far end. Without it a simulated map is
                # suspiciously crisp, and every filter downstream goes untested.
                sigma = 0.003 + 0.02 * (at / MAX_RANGE)
                measured = max(0.0, at + self.random.normal(0.0, sigma * self.noise))
                distance_mm[zone * TARGETS + slot] = int(measured * 1000.0)
                status[zone * TARGETS + slot] = STATUS_MERGED if merged > 1 else STATUS_VALID
        return distance_mm, status
