"""Exercise MuJoCo sensors and the body wire; no training stack or GPU required."""

import json
import socket
import threading

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import DEFAULT_SCENE, Body, Handler, Server, World
from mjlab_microduck.sim.head_imu import ACCEL_LSB, GYRO_LSB, G, HeadImu
from mjlab_microduck.sim.tof import Tof


def sensor_model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.005" gravity="0 0 -9.80665"/><worldbody>
    <geom type="plane" size="10 10 .1"/>
    <body pos="0 0 1"><freejoint/><geom type="sphere" size=".01" mass="1"/>
    <site name="head_imu" pos=".1 0 0"/><site name="tof" pos=".1 0 0"/>
    </body></worldbody><sensor><gyro name="head_gyro" site="head_imu"/>
    <accelerometer name="head_accel" site="head_imu"/></sensor></mujoco>""")


def test_chip_axes_gravity_spin_and_saturation():
    model = sensor_model()
    data = mujoco.MjData(model)
    # Forward the sensor pass with zero acceleration: a supported chip sees +g.
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_sensorAcc(model, data)
    imu = HeadImu(model)
    imu.capture(data, 123)
    frame = imu.since(0)["samples"][0]
    np.testing.assert_allclose(frame["accel"], [0, 0, G], atol=ACCEL_LSB)
    assert frame["t_ns"] == 123
    # Rotate the chip 90 degrees around Y. Gravity must rotate into chip -X.
    data.qpos[3:7] = [2**-0.5, 0, 2**-0.5, 0]
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_sensorAcc(model, data)
    imu.capture(data, 456)
    np.testing.assert_allclose(imu.samples[-1]["accel"], [-G, 0, 0], atol=ACCEL_LSB)
    data.qvel[3:] = [0, 0, 2]
    mujoco.mj_forward(model, data)
    imu.capture(data, 789)
    np.testing.assert_allclose(imu.samples[-1]["gyro"], [0, 0, 2], atol=GYRO_LSB)
    data.qvel[3:] = [0, 0, 100]
    mujoco.mj_forward(model, data)
    imu.capture(data, 900)
    assert imu.samples[-1]["gyro"][2] == pytest.approx(32767 * GYRO_LSB)


def test_noise_is_seeded_and_independent_of_readers():
    model = sensor_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    a, b = (HeadImu(model, noise=1, seed=42) for _ in range(2))
    for tick in range(600):
        a.capture(data, tick)
        a.since(0)  # A subscriber cannot perturb the sensor's random sequence.
        b.capture(data, tick)
    assert a.since(0) == b.since(0)
    assert len(a.samples) == 512
    assert np.std([s["accel"][0] for s in a.samples]) > ACCEL_LSB
    for ratio in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            HeadImu(model, noise=ratio)
        with pytest.raises(ValueError):
            Tof(model, model.site("tof").id, noise=ratio)


def test_wire_reports_version_mismatch_and_preserves_physics_samples():
    world = World(DEFAULT_SCENE)
    world.epoch_ns -= 1_000_000_000
    body = Body(world, 0)
    body.place(None, 0.125, 0)
    world.bodies.append(body)
    server = Server(("127.0.0.1", 0), Handler)
    server.body = body
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with (
            socket.create_connection(server.server_address) as sock,
            sock.makefile("rw") as wire,
        ):

            def call(request):
                wire.write(json.dumps(request) + "\n")
                wire.flush()
                return json.loads(wire.readline())

            assert "protocol 1" in call({"op": "hello", "protocol": 1})["error"]
            assert call({"op": "hello", "protocol": 2})["protocol"] == 2
            world.step(4)
            samples = call({"op": "head_imu", "after": 0})["samples"]
            assert len(samples) == 4
            assert np.diff([s["t_ns"] for s in samples]).tolist() == [5_000_000] * 3
            assert call({"op": "head_imu", "after": 4})["samples"] == []
            depth = call({"op": "tof"})
            assert (depth["rows"], depth["cols"]) == (8, 8)
            assert len(depth["distance_mm"]) == len(depth["status"]) == 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_exact_tof_measures_plane():
    model = sensor_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tof = Tof(model, model.site("tof").id, noise=0)
    distances, status = tof.frame(data)
    # Downward rows see the floor at height 1 m, along the slanted ray.
    for i, direction in enumerate(tof.directions):
        if direction[2] < -0.25:
            assert status[i] == 5
            assert distances[i] == pytest.approx(int(-1000 / direction[2]), abs=1)


def test_beta_game_rotation_rate_and_seeded_chip_noise():
    from mjlab_microduck.sim.head_imu import BetaHeadImu

    model = sensor_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_sensorAcc(model, data)
    imu = BetaHeadImu(model)
    # Analytic chip motion, not orientation truth: one rad/s about gravity.
    data.sensordata[model.sensor("head_gyro").adr[0] :][:3] = [0, 0, 1]
    for tick in range(401):
        imu.capture(data, 1_000_000_000 + tick * 5_000_000)
    samples = imu.since(0)["samples"]
    assert 120 <= len(samples) <= 121
    np.testing.assert_allclose(np.diff([s["t_ns"] for s in samples]), 1e9 / 60, atol=1)
    q = samples[-1]["quat"]
    assert abs(2 * np.arctan2(q[3], q[0]) - 2) < 0.02
    a, b = (BetaHeadImu(model, noise=1, seed=17) for _ in range(2))
    for tick in range(100):
        a.capture(data, tick * 5_000_000)
        b.capture(data, tick * 5_000_000)
    assert a.since(0) == b.since(0)


def test_beta_world_uses_face_axes_and_usable_l8_range():
    world = World(DEFAULT_SCENE, board="beta")
    body = Body(world, 0)
    body.place(None, 0.125, 0)
    world.bodies.append(body)
    world.step(20)
    assert body.tof.minimum == 0.12 and body.tof.maximum == 3.5
    assert body.head_imu.samples[-1]["quat"]
    force = body.head_imu.samples[-1]["accel"]
    assert 9 < np.linalg.norm(force) < 11
    head_force = np.zeros(3)
    mujoco.mju_rotVecQuat(head_force, np.array(force), np.array([.5, .5, -.5, -.5]))
    np.testing.assert_allclose(head_force, [0, 0, 9.81], atol=.02)
