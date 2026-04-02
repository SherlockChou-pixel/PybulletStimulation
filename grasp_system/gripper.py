import numpy as np
import pybullet as p


class GripperController:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.current_grip_force = 0.0
        self.current_grip_target = self.config.gripper.open_position

    @staticmethod
    def _fmt_vec(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    @staticmethod
    def _to_float(value, default=0.0):
        if isinstance(value, (int, float, np.number)):
            return float(value)
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return float(arr[0])
        except Exception:
            pass
        return float(default)

    @property
    def panda_id(self):
        return self.runtime.panda_id

    @property
    def object_id(self):
        return self.runtime.obj_id

    @property
    def gripper_joints(self):
        return self.config.gripper.joints

    def set_gripper(self, target_pos, force):
        target_pos = float(np.clip(target_pos, 0.0, self.config.gripper.open_position))
        force = float(max(0.0, force))
        self.current_grip_target = target_pos
        self.current_grip_force = force
        for joint_id in self.gripper_joints:
            p.setJointMotorControl2(self.panda_id, joint_id, p.POSITION_CONTROL, target_pos, force=force)

    def open_gripper(self, target_pos=None, force=None):
        if target_pos is None:
            target_pos = self.config.gripper.open_position
        if force is None:
            force = self.config.gripper.open_force
        self.set_gripper(target_pos, force)
        self.logger.info('gripper_open target_pos=%.4f force=%.2f', target_pos, force)

    def close_gripper(self, target_pos=None, force=None):
        if target_pos is None:
            target_pos = self.config.gripper.default_close_position
        if force is None:
            force = self.config.gripper.close_force
        self.set_gripper(target_pos, force)
        self.logger.info('gripper_close target_pos=%.4f force=%.2f', target_pos, force)

    def maintain_grip(self):
        self.set_gripper(self.current_grip_target, self.current_grip_force)

    def reinforce_grip_for_transport(self, multiplier=None, force_cap=None):
        if multiplier is None:
            multiplier = self.config.gripper.carry_force_multiplier
        if force_cap is None:
            force_cap = self.config.gripper.carry_force_cap
        previous_force = float(self.current_grip_force)
        boosted_force = min(previous_force * float(multiplier), float(force_cap))
        boosted_force = max(boosted_force, previous_force)
        self.set_gripper(self.current_grip_target, boosted_force)
        self.logger.info(
            'GRIP_TRANSPORT reinforce target_pos=%.4f previous_force=%.2f boosted_force=%.2f',
            self.current_grip_target,
            previous_force,
            boosted_force,
        )
        return boosted_force

    def estimate_gripper_force(self, object_id=None):
        if object_id is None:
            object_id = self.object_id

        mass = 0.1
        try:
            mass = p.getDynamicsInfo(object_id, -1)[0]
        except Exception:
            self.logger.exception('estimate_gripper_force failed to read dynamics, fallback to default mass')

        if mass <= 0:
            mass = 0.1

        aabb_min, aabb_max = p.getAABB(object_id)
        object_size = np.array(aabb_max) - np.array(aabb_min)
        horizontal_span = max(0.01, float(min(object_size[0], object_size[1])))
        finger_target_pos = np.clip(horizontal_span * 0.5 - 0.003, 0.005, self.config.gripper.open_position)

        gravity_force = mass * 9.8
        size_factor = np.clip(horizontal_span / 0.05, 0.6, 1.6)
        target_torque = np.clip(0.035 + gravity_force * 0.008 + (size_factor - 1.0) * 0.010, 0.030, 0.080)
        max_motor_force = np.clip(
            target_torque * 1.20,
            self.config.gripper.motor_force_min / self.config.gripper.force_scale,
            0.100,
        )
        hold_force = np.clip(
            target_torque * 1.05,
            self.config.gripper.hold_force_min / self.config.gripper.force_scale,
            0.090,
        )

        return {
            'mass': mass,
            'object_size': object_size,
            'target_torque': float(target_torque),
            'max_motor_force': float(max_motor_force),
            'hold_force': float(hold_force),
            'finger_target_pos': float(finger_target_pos),
        }

    def get_contact_state(self, object_id=None):
        if object_id is None:
            object_id = self.object_id

        contact_count = 0
        total_normal_force = 0.0
        total_lateral_friction = 0.0
        total_spinning_friction = 0.0
        active_fingers = 0
        friction_coefficients = []

        for joint_id in self.gripper_joints:
            contacts = p.getContactPoints(bodyA=self.panda_id, bodyB=object_id, linkIndexA=joint_id)
            if contacts:
                active_fingers += 1
            for contact in contacts:
                total_normal_force += contact[9]
                if len(contact) > 13:
                    friction_coefficients.append(contact[13])
                if len(contact) > 10:
                    total_lateral_friction += contact[10]
                if len(contact) > 12:
                    total_lateral_friction += contact[12]
                if len(contact) > 14:
                    total_spinning_friction += contact[14]
                contact_count += 1

        avg_friction_coeff = np.mean(friction_coefficients) if friction_coefficients else 0.0
        return {
            'contact_count': contact_count,
            'active_fingers': active_fingers,
            'total_normal_force': float(total_normal_force),
            'total_lateral_friction': float(total_lateral_friction),
            'total_spinning_friction': float(total_spinning_friction),
            'avg_friction_coefficient': float(avg_friction_coeff),
            'friction_coefficients': friction_coefficients,
        }

    def log_contact_state(self, stage, object_id=None):
        state = self.get_contact_state(object_id)
        self.logger.info(
            'GRIP_CONTACT stage=%s contact_count=%d active_fingers=%d normal_force=%.4f lateral_friction=%.4f spinning_friction=%.4f avg_mu=%.3f grip_target=%.4f grip_force=%.2f',
            stage,
            state['contact_count'],
            state['active_fingers'],
            state['total_normal_force'],
            state['total_lateral_friction'],
            state['total_spinning_friction'],
            state['avg_friction_coefficient'],
            self.current_grip_target,
            self.current_grip_force,
        )
        return state

    def calculate_min_grasp_force(self, object_id=None):
        if object_id is None:
            object_id = self.object_id

        try:
            dynamics_info = p.getDynamicsInfo(object_id, -1)
            mass = self._to_float(dynamics_info[0], 0.1)
            lateral_friction = self._to_float(dynamics_info[1], 1.0)
            gravity = 9.8
            weight = mass * gravity
            safe_mu = max(float(lateral_friction), 0.1)
            min_normal_force = weight / safe_mu
            min_grasp_force = min_normal_force * 1.2
            min_grasp_force = max(min_grasp_force, self.config.gripper.min_grasp_force)
            self.logger.info(
                'GRIP_FORCE mass=%.3f mu=%.3f computed_force=%.3f applied_floor=%.3f',
                mass,
                safe_mu,
                min_normal_force * 1.2,
                min_grasp_force,
            )
            return min_grasp_force
        except Exception as exc:
            self.logger.warning('calculate_min_grasp_force failed: %s', str(exc))
            return float(max(self.config.gripper.close_force, self.config.gripper.min_grasp_force))

    def close_with_fixed_force(self, object_id=None, fixed_force=None, close_position=None, stabilize_steps=None):
        if object_id is None:
            object_id = self.object_id

        estimate = self.estimate_gripper_force(object_id)
        if fixed_force is None:
            fixed_force = self.calculate_min_grasp_force(object_id)
        fixed_force = float(max(fixed_force, self.config.gripper.min_grasp_force))

        if close_position is None:
            close_position = max(
                min(float(estimate['finger_target_pos']), self.config.gripper.open_position),
                self.config.gripper.default_close_position,
            )
        if stabilize_steps is None:
            stabilize_steps = self.config.motion.close_stabilize_steps

        self.logger.info(
            'GRIP_EXECUTE close_position=%.4f force=%.2f estimate_size=%s finger_target=%.4f',
            close_position,
            fixed_force,
            self._fmt_vec(estimate['object_size']),
            estimate['finger_target_pos'],
        )

        self.log_contact_state('before_close_command', object_id)

        already_open = abs(self.current_grip_target - self.config.gripper.open_position) < 1e-4
        if not already_open:
            self.open_gripper()
            self.runtime.step(30)
            self.log_contact_state('after_open', object_id)

        self.set_gripper(close_position, fixed_force)
        self.logger.info('GRIP_CLOSE_COMMAND target_pos=%.4f force=%.2f', close_position, fixed_force)

        for _ in range(40):
            self.runtime.step(1)

        pre_contact_state = self.log_contact_state('after_initial_close', object_id)

        for _ in range(stabilize_steps):
            self.set_gripper(close_position, fixed_force)
            self.runtime.step(1)

        contact_state = self.log_contact_state('after_stabilize', object_id)

        try:
            dynamics_info = p.getDynamicsInfo(object_id, -1)
            object_mass = self._to_float(dynamics_info[0], 0.0)
            lateral_friction = self._to_float(dynamics_info[1], 0.0)
            spinning_friction = self._to_float(dynamics_info[3], 0.0) if len(dynamics_info) > 3 else 0.0
            rolling_friction = self._to_float(dynamics_info[4], 0.0) if len(dynamics_info) > 4 else 0.0
            self.logger.info(
                'OBJECT_DYNAMICS mass=%.3f lateral_mu=%.3f spinning=%.4f rolling=%.4f',
                object_mass,
                lateral_friction,
                spinning_friction,
                rolling_friction,
            )
        except Exception as exc:
            self.logger.warning('OBJECT_DYNAMICS unavailable: %s', str(exc))

        return {
            'final_pos': close_position,
            'hold_force': fixed_force,
            'pre_contact_state': pre_contact_state,
            'contact_state': contact_state,
        }
