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

    def connect(self, gui: bool = True):
        if self.connected:
            return

        connection_mode = p.GUI if gui else p.DIRECT
        p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
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
        self.logger.info('simulation_ready robot_id=%s object_id=%s', self.panda_id, self.obj_id)

    def step(self, steps: int = 1, hook=None):
        realtime_sleep = self.config.runtime.realtime_sleep
        sleep_dt = self.config.runtime.sleep_dt
        for _ in range(steps):
            if hook is not None:
                hook()
            p.stepSimulation()
            if realtime_sleep and sleep_dt > 0:
                time.sleep(sleep_dt)

    def disconnect(self):
        if self.connected:
            p.disconnect()
            self.connected = False
            self.logger.info('simulation_disconnected')
