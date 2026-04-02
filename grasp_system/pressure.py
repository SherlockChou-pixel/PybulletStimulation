from collections import deque

import numpy as np
import pybullet as p


class PressureSystem:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.history = deque(maxlen=self.config.pressure.history_length)
        self._active = False

    @staticmethod
    def _as_list(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    def activate(self):
        if self._active or not self.config.pressure.enabled:
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
        self.history = deque(maxlen=self.config.pressure.history_length)
        if self.runtime.connected:
            self.step_update(self.runtime.sim_time, self.runtime.sim_step_count)

    def _finger_name(self, index, joint_id):
        if index == 0:
            return 'left_finger'
        if index == 1:
            return 'right_finger'
        return f'finger_{joint_id}'

    def _summarize_contacts(self, contacts):
        if not contacts:
            return {
                'contact_count': 0,
                'total_normal_force': 0.0,
                'total_lateral_force': 0.0,
                'max_normal_force': 0.0,
                'center_of_pressure': [0.0, 0.0, 0.0],
                'pressure_proxy': 0.0,
            }

        normal_forces = [float(contact[9]) for contact in contacts if len(contact) > 9]
        lateral_forces = [
            float(abs(contact[10])) + float(abs(contact[12]))
            for contact in contacts
            if len(contact) > 12
        ]
        contact_positions = [np.array(contact[5], dtype=float) for contact in contacts if len(contact) > 5]
        center_of_pressure = np.mean(contact_positions, axis=0) if contact_positions else np.zeros(3, dtype=float)
        total_normal_force = float(np.sum(normal_forces)) if normal_forces else 0.0

        return {
            'contact_count': len(contacts),
            'total_normal_force': total_normal_force,
            'total_lateral_force': float(np.sum(lateral_forces)) if lateral_forces else 0.0,
            'max_normal_force': float(max(normal_forces)) if normal_forces else 0.0,
            'center_of_pressure': self._as_list(center_of_pressure),
            'pressure_proxy': float(total_normal_force / max(1, len(contacts))),
        }

    def sample(self, object_id=None):
        if not self.runtime.connected:
            return None
        if object_id is None:
            object_id = self.runtime.obj_id

        per_finger = {}
        total_normal_force = 0.0
        total_lateral_force = 0.0
        active_fingers = 0

        for index, joint_id in enumerate(self.config.gripper.joints):
            contacts = p.getContactPoints(
                bodyA=self.runtime.panda_id,
                bodyB=object_id,
                linkIndexA=joint_id,
            )
            metrics = self._summarize_contacts(contacts)
            per_finger[self._finger_name(index, joint_id)] = metrics
            total_normal_force += metrics['total_normal_force']
            total_lateral_force += metrics['total_lateral_force']
            if metrics['contact_count'] > 0:
                active_fingers += 1

        stable_contact = (
            active_fingers >= self.config.pressure.min_active_fingers
            and total_normal_force >= self.config.pressure.min_total_normal_force
        )
        return {
            'time': float(self.runtime.sim_time),
            'active_fingers': active_fingers,
            'total_normal_force': float(total_normal_force),
            'total_lateral_force': float(total_lateral_force),
            'stable_contact': bool(stable_contact),
            'per_finger': per_finger,
        }

    def step_update(self, sim_time=None, step_index=None):
        if not self.config.pressure.enabled:
            return
        if step_index is None:
            step_index = self.runtime.sim_step_count
        sample_every_steps = max(1, int(self.config.pressure.sample_every_steps))
        if step_index % sample_every_steps != 0:
            return
        sample = self.sample()
        if sample is None:
            return
        if sim_time is not None:
            sample['time'] = float(sim_time)
        self.history.append(sample)

        if self.config.pressure.log_samples:
            self.logger.info(
                'PRESSURE_SAMPLE time=%.4f active_fingers=%d total_normal_force=%.4f total_lateral_force=%.4f stable=%s',
                sample['time'],
                sample['active_fingers'],
                sample['total_normal_force'],
                sample['total_lateral_force'],
                sample['stable_contact'],
            )

    def get_latest(self):
        if not self.history:
            return self.sample()
        return self.history[-1]

    def build_summary(self, window_size=None):
        if not self.history:
            latest = self.sample()
            if latest is None:
                return None
            window = [latest]
        else:
            if window_size is None:
                window_size = self.config.pressure.summary_window
            window = list(self.history)[-max(1, int(window_size)) :]

        latest = window[-1]
        normal_force_series = [item['total_normal_force'] for item in window]
        lateral_force_series = [item['total_lateral_force'] for item in window]
        stable_ratio = sum(1 for item in window if item['stable_contact']) / len(window)

        return {
            'time': latest['time'],
            'active_fingers': latest['active_fingers'],
            'total_normal_force': latest['total_normal_force'],
            'total_lateral_force': latest['total_lateral_force'],
            'stable_contact': latest['stable_contact'],
            'peak_normal_force': max(normal_force_series),
            'mean_normal_force': float(np.mean(normal_force_series)),
            'mean_lateral_force': float(np.mean(lateral_force_series)),
            'stable_ratio': float(stable_ratio),
            'per_finger': latest['per_finger'],
            'sample_count': len(window),
        }
