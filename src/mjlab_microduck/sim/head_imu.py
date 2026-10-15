"""BMI088 samples at the CAD HAT site; orientation is fused by tofd, never truth.

Scales match bmi088-rs v0.1.2 Config::default (3 g, 500 dps). Noise/offset:
Bosch BST-BMI088-DS000-19 rev 1.9, tables 4–6, pp. 9–11:
https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf
The 60 s correlated bias time is a simulation choice, not a datasheet guarantee.
"""

from collections import deque

import numpy as np

G = 9.80665
ACCEL_LSB = G / 10920.0
GYRO_LSB = np.radians(1.0 / 65.536)
# Datasheet noise densities; the 100 Hz ODR settings filter gyro at 32 Hz
# and accel at approximately 40 Hz (normal mode).
DENSITY = np.array([np.radians(0.014)] * 3 + [160e-6 * G, 160e-6 * G, 190e-6 * G])
BANDWIDTH = np.array([32.0] * 3 + [40.0] * 3)
OFFSET = np.array([np.radians(1.0)] * 3 + [0.020 * G] * 3)


class HeadImu:
    def __init__(self, model, prefix="", noise=0.0, seed=0):
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("IMU noise ratio must be finite and nonnegative")
        self.dt = float(model.opt.timestep)
        self.noise = noise
        self.random = np.random.default_rng(seed)
        self.bias = self.random.uniform(-OFFSET, OFFSET)
        self.samples = deque(maxlen=512)
        self.sequence = 0
        self.sensors = [
            model.sensor(prefix + name).adr[0] for name in ("head_gyro", "head_accel")
        ]

    def capture(self, data, t_ns):
        # Bounded, correlated slow bias, independent of requests and subscriber count.
        decay = np.exp(-self.dt / 60.0)
        self.bias = decay * self.bias + np.sqrt(1 - decay**2) * self.random.normal(
            0, OFFSET / 3
        )
        values = np.concatenate([data.sensordata[a : a + 3] for a in self.sensors])
        values = values + self.noise * (
            self.bias + self.random.normal(0, DENSITY * np.sqrt(BANDWIDTH))
        )
        lsb = np.array([GYRO_LSB] * 3 + [ACCEL_LSB] * 3)
        values = np.clip(np.rint(values / lsb), -32768, 32767) * lsb
        self.sequence += 1
        self.samples.append(
            {
                "seq": self.sequence,
                "t_ns": t_ns,
                "gyro": values[:3].tolist(),
                "accel": values[3:].tolist(),
                "temp_c": 32.0,
            }
        )

    def since(self, after, until_ns=None):
        return {
            "samples": [
                sample
                for sample in self.samples
                if sample["seq"] > after
                and (until_ns is None or sample["t_ns"] <= until_ns)
            ]
        }
