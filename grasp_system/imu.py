from collections import deque

import numpy as np
import pybullet as p

from .config import PANDA_END_EFFECTOR_INDEX


class IMUSystem:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.history = {}
        self._previous_state = {}
        self._active = False

    @staticmethod
    def _as_list(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    def _tracked_targets(self):
        targets = []
        if self.config.imu.track_end_effector:
            targets.append('end_effector')
        if self.config.imu.track_object:
            targets.append('object')
        return targets

    def activate(self):
        if self._active or not self.config.imu.enabled:
            return
        self.reset()
        self.runtime.register_step_callback(self.step_update)
        self._active = True

    def deactivate(self):
        if not self._active:
            return
        self.runtime.unregister_step_callback(self.step_update)
        self._active = False

    def reset(self):
        history_length = self.config.imu.history_length
        self.history = {target: deque(maxlen=history_length) for target in self._tracked_targets()}
        self._previous_state = {}
        if self.runtime.connected:
            self.step_update(self.runtime.sim_time, self.runtime.sim_step_count, bootstrap=True)

    def _capture_end_effector_state(self):
        link_state = p.getLinkState(
            self.runtime.panda_id,
            PANDA_END_EFFECTOR_INDEX,
            computeForwardKinematics=1,
            computeLinkVelocity=1,
        )
        return {
            'position': np.array(link_state[4], dtype=float),
            'orientation': np.array(link_state[5], dtype=float),
            'linear_velocity': np.array(link_state[6], dtype=float),
            'angular_velocity': np.array(link_state[7], dtype=float),
        }

    def _capture_object_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.runtime.obj_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.runtime.obj_id)
        return {
            'position': np.array(position, dtype=float),
            'orientation': np.array(orientation, dtype=float),
            'linear_velocity': np.array(linear_velocity, dtype=float),
            'angular_velocity': np.array(angular_velocity, dtype=float),
        }

    def _capture_state(self, target):
        if not self.runtime.connected:
            return None
        if target == 'end_effector':
            return self._capture_end_effector_state()
        if target == 'object':
            return self._capture_object_state()
        return None

    def step_update(self, sim_time=None, step_index=None, bootstrap=False):
        if not self.runtime.connected or not self.config.imu.enabled:
            return
        if sim_time is None:
            sim_time = self.runtime.sim_time
        if step_index is None:
            step_index = self.runtime.sim_step_count
        sample_every_steps = max(1, int(self.config.imu.sample_every_steps))
        if not bootstrap and step_index % sample_every_steps != 0:
            return

        for target in self._tracked_targets():
            state = self._capture_state(target)
            if state is None:
                continue

            previous = self._previous_state.get(target)
            if previous is None or bootstrap:
                linear_acceleration = np.zeros(3, dtype=float)
                angular_acceleration = np.zeros(3, dtype=float)
            else:
                dt = max(float(sim_time - previous['time']), 1e-6)
                linear_acceleration = (state['linear_velocity'] - previous['linear_velocity']) / dt
                angular_acceleration = (state['angular_velocity'] - previous['angular_velocity']) / dt

            sample = {
                'time': float(sim_time),
                'position': state['position'],
                'orientation': state['orientation'],
                'linear_velocity': state['linear_velocity'],
                'angular_velocity': state['angular_velocity'],
                'linear_acceleration': linear_acceleration,
                'angular_acceleration': angular_acceleration,
            }
            self.history[target].append(sample)
            self._previous_state[target] = sample

            if self.config.imu.log_samples:
                self.logger.info(
                    'IMU_SAMPLE target=%s time=%.4f position=%s linear_velocity=%s linear_acc=%s angular_velocity=%s',
                    target,
                    sample['time'],
                    self._as_list(sample['position']),
                    self._as_list(sample['linear_velocity']),
                    self._as_list(sample['linear_acceleration']),
                    self._as_list(sample['angular_velocity']),
                )

    def get_latest(self, target):
        samples = self.history.get(target)
        if not samples:
            return None
        sample = samples[-1]
        return {
            'time': sample['time'],
            'position': self._as_list(sample['position']),
            'orientation': self._as_list(sample['orientation']),
            'linear_velocity': self._as_list(sample['linear_velocity']),
            'angular_velocity': self._as_list(sample['angular_velocity']),
            'linear_acceleration': self._as_list(sample['linear_acceleration']),
            'angular_acceleration': self._as_list(sample['angular_acceleration']),
        }

    def build_summary(self, target, window_size=None):
        samples = self.history.get(target)
        if not samples:
            return None

        if window_size is None:
            window_size = self.config.imu.summary_window
        window = list(samples)[-max(1, int(window_size)) :]
        latest = window[-1]

        linear_speeds = [float(np.linalg.norm(item['linear_velocity'])) for item in window]
        angular_speeds = [float(np.linalg.norm(item['angular_velocity'])) for item in window]
        linear_accels = [float(np.linalg.norm(item['linear_acceleration'])) for item in window]
        angular_accels = [float(np.linalg.norm(item['angular_acceleration'])) for item in window]

        return {
            'time': latest['time'],
            'latest_position': self._as_list(latest['position']),
            'latest_linear_speed': linear_speeds[-1],
            'latest_angular_speed': angular_speeds[-1],
            'latest_linear_acceleration': linear_accels[-1],
            'peak_linear_acceleration': max(linear_accels),
            'peak_angular_acceleration': max(angular_accels),
            'mean_linear_speed': float(np.mean(linear_speeds)),
            'mean_angular_speed': float(np.mean(angular_speeds)),
            'sample_count': len(window),
        }
