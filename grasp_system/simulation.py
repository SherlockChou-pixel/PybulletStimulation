import time

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
        self.obj_id = p.loadURDF(scene.object_urdf, scene.object_position)
        self.connected = True
        self.sim_time = 0.0
        self.sim_step_count = 0
        self.logger.info('simulation_ready robot_id=%s object_id=%s', self.panda_id, self.obj_id)

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
            self.logger.info('simulation_disconnected')
