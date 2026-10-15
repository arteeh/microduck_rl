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


class BetaHeadImu(HeadImu):
    """LSM6DSV16X game rotation: gravity/gyro complementary fusion, no magnetic yaw.

    ST DS13510 rev 4, table 3: 2.8 mdps/sqrtHz, 60 ug/sqrtHz,
    +/-1 dps and +/-12 mg offset; 500 dps/4g scales match robotd CONFIG.
    SFLP internals are not public: the 2 s tilt correction and 60 s bias drift
    are explicit simulation assumptions, not a bit-exact SFLP implementation.
    https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf
    """

    def __init__(self, model, prefix="", noise=0.0, seed=0):
        super().__init__(model, prefix, noise, seed)
        self.density = np.array([np.radians(0.0028)] * 3 + [60e-6 * G] * 3)
        self.offset = np.array([np.radians(1.0)] * 3 + [0.012 * G] * 3)
        self.bias = self.random.uniform(-self.offset, self.offset)
        self.previous = None
        self.next_sample = None
        self.quat = None
        self.filtered_accel = None

    def capture(self, data, t_ns):
        import mujoco

        decay = np.exp(-self.dt / 60)
        self.bias = decay * self.bias + np.sqrt(1 - decay**2) * self.random.normal(
            0, self.offset / 3
        )
        values = np.concatenate([data.sensordata[a : a + 3] for a in self.sensors])
        values += self.noise * (
            self.bias + self.random.normal(0, self.density * np.sqrt(60))
        )
        lsb = np.array([np.radians(0.0175)] * 3 + [0.000122 * G] * 3)
        # LSM6 full-scale is 500 dps/4g; signed ADC has additional headroom.
        limits = np.array([np.radians(500)] * 3 + [4 * G] * 3)
        values = np.rint(np.clip(values, -limits, limits) / lsb) * lsb
        if self.previous is None:
            self.previous = t_ns, values.copy()
            self.next_sample = float(t_ns)
        old_time, old = self.previous
        while self.next_sample <= t_ns:
            alpha = (self.next_sample - old_time) / max(t_ns - old_time, 1)
            sample = old + np.clip(alpha, 0, 1) * (values - old)
            # Sampling a fractional physics tick must still return integer ADC counts.
            sample = np.clip(np.rint(sample / lsb), -32768, 32767) * lsb
            gyro, accel = sample[:3], sample[3:]
            magnitude = np.linalg.norm(accel)
            if self.quat is None:
                self.quat = np.array([1.0, 0, 0, 0])
                if magnitude > 1e-6:
                    measured = accel / magnitude
                    axis = np.cross(measured, [0, 0, 1])
                    q = np.r_[1 + measured[2], axis]
                    self.quat = (
                        q / np.linalg.norm(q)
                        if np.linalg.norm(q) > 1e-6
                        else np.array([0.0, 1, 0, 0])
                    )
                self.filtered_accel = accel.copy()
            else:
                expected = np.zeros(3)
                inverse = self.quat.copy()
                inverse[1:] *= -1
                mujoco.mju_rotVecQuat(expected, np.array([0.0, 0, 1]), inverse)
                correction = (
                    np.cross(accel / magnitude, expected) * 0.5
                    if abs(magnitude - G) < 0.2 * G
                    else np.zeros(3)
                )
                mujoco.mju_quatIntegrate(self.quat, gyro + correction, 1 / 60)
                self.filtered_accel += (1 - np.exp(-2 * np.pi * 6 / 60)) * (
                    accel - self.filtered_accel
                )
            q = self.quat if self.quat[0] >= 0 else -self.quat
            xyz = q[1:].astype(np.float16).astype(float)
            fused = [np.sqrt(max(0, 1 - np.dot(xyz, xyz))), *xyz]
            self.sequence += 1
            self.samples.append(
                {
                    "seq": self.sequence,
                    "t_ns": round(self.next_sample),
                    "gyro": gyro.tolist(),
                    "accel": (
                        np.clip(np.rint(self.filtered_accel / lsb[3:]), -32768, 32767)
                        * lsb[3:]
                    ).tolist(),
                    "quat": fused,
                    "temp_c": 32.0,
                }
            )
            self.next_sample += 1e9 / 60
        self.previous = t_ns, values.copy()
