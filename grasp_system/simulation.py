import time

import numpy as np
import pybullet as p
import pybullet_data

from .config import AppConfig


class SimulationRuntime:
    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger
        self.panda_id = None
        self.obj_id = None
        self.land_id = None
        self.connected = False
        self.sim_time = 0.0
        self.sim_step_count = 0
        self.step_callbacks = []
        self.object_spawn_position = None
        self.object_spawn_orientation = None

    def _build_rng(self):
        return np.random.default_rng(self.config.scene.random_seed)

    def _sample_object_position(self, rng):
        scene = self.config.scene
        if not scene.randomize_object_position:
            return np.array(scene.object_position, dtype=float)

        x = float(rng.uniform(*scene.object_x_range))
        y = float(rng.uniform(*scene.object_y_range))
        z = float(scene.object_spawn_z)
        return np.array([x, y, z], dtype=float)

    def _sample_object_orientation(self, rng):
        scene = self.config.scene
        roll, pitch, yaw = scene.object_orientation_euler
        if scene.randomize_object_yaw:
            yaw = float(rng.uniform(*scene.object_yaw_range))
        orientation_euler = np.array([roll, pitch, yaw], dtype=float)
        return orientation_euler, p.getQuaternionFromEuler(orientation_euler.tolist())

    def _apply_scene_dynamics(self):
        scene = self.config.scene

        if self.land_id is not None:
            plane_kwargs = {}
            if scene.floor_lateral_friction is not None:
                plane_kwargs["lateralFriction"] = float(scene.floor_lateral_friction)
            if scene.floor_spinning_friction is not None:
                plane_kwargs["spinningFriction"] = float(scene.floor_spinning_friction)
            if scene.floor_rolling_friction is not None:
                plane_kwargs["rollingFriction"] = float(scene.floor_rolling_friction)
            if plane_kwargs:
                p.changeDynamics(self.land_id, -1, **plane_kwargs)

        if self.obj_id is not None:
            object_kwargs = {}
            if scene.object_mass is not None:
                object_kwargs["mass"] = float(scene.object_mass)
            if scene.object_lateral_friction is not None:
                object_kwargs["lateralFriction"] = float(scene.object_lateral_friction)
            if scene.object_spinning_friction is not None:
                object_kwargs["spinningFriction"] = float(scene.object_spinning_friction)
            if scene.object_rolling_friction is not None:
                object_kwargs["rollingFriction"] = float(scene.object_rolling_friction)
            if scene.object_restitution is not None:
                object_kwargs["restitution"] = float(scene.object_restitution)
            if scene.object_linear_damping is not None:
                object_kwargs["linearDamping"] = float(scene.object_linear_damping)
            if scene.object_angular_damping is not None:
                object_kwargs["angularDamping"] = float(scene.object_angular_damping)
            if object_kwargs:
                p.changeDynamics(self.obj_id, -1, **object_kwargs)

    def connect(self, gui: bool = True):
        if self.connected:
            return

        connection_mode = p.GUI if gui else p.DIRECT
        p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.runtime.sim_dt)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=60,
            cameraPitch=-25,
            cameraTargetPosition=[0.6, 0.0, 0.3],
        )

        scene = self.config.scene
        self.land_id = p.loadURDF(scene.plane_urdf, [0, 0, 0])
        self.panda_id = p.loadURDF(scene.robot_urdf, scene.robot_base_position, useFixedBase=scene.use_fixed_base)
        rng = self._build_rng()
        self.object_spawn_position = self._sample_object_position(rng)
        object_euler, object_quat = self._sample_object_orientation(rng)
        self.object_spawn_orientation = np.array(object_quat, dtype=float)
        self.obj_id = p.loadURDF(scene.object_urdf, self.object_spawn_position, object_quat)
        self._apply_scene_dynamics()
        self.connected = True
        self.sim_time = 0.0
        self.sim_step_count = 0
        self.logger.info(
            'simulation_ready scenario=%s robot_id=%s object_id=%s object_spawn=%s object_euler=%s randomized=%s object_mass=%s object_mu=%s floor_mu=%s',
            scene.scenario_name,
            self.panda_id,
            self.obj_id,
            np.round(self.object_spawn_position, 4).tolist(),
            np.round(object_euler, 4).tolist(),
            scene.randomize_object_position,
            None if scene.object_mass is None else round(float(scene.object_mass), 4),
            None if scene.object_lateral_friction is None else round(float(scene.object_lateral_friction), 4),
            None if scene.floor_lateral_friction is None else round(float(scene.floor_lateral_friction), 4),
        )

    def register_step_callback(self, callback):
        if not any(existing == callback for existing in self.step_callbacks):
            self.step_callbacks.append(callback)

    def unregister_step_callback(self, callback):
        self.step_callbacks = [item for item in self.step_callbacks if item != callback]

    def step(self, steps: int = 1, hook=None):
        realtime_sleep = self.config.runtime.realtime_sleep
        sleep_dt = self.config.runtime.sleep_dt
        for _ in range(steps):
            if hook is not None:
                hook()
            p.stepSimulation()
            self.sim_time += self.config.runtime.sim_dt
            self.sim_step_count += 1
            for callback in list(self.step_callbacks):
                callback(self.sim_time, self.sim_step_count)
            if realtime_sleep and sleep_dt > 0:
                time.sleep(sleep_dt)

    def disconnect(self):
        if self.connected:
            p.disconnect()
            self.connected = False
            self.sim_time = 0.0
            self.sim_step_count = 0
            self.object_spawn_position = None
            self.object_spawn_orientation = None
            self.logger.info('simulation_disconnected')
