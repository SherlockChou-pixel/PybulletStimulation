import numpy as np
import pybullet as p


class GripperController:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.current_grip_force = 0.0
        self.current_grip_target = self.config.gripper.open_position

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
        self.logger.info("夹爪打开 target_pos=%.4f force=%.2f", target_pos, force)

    def close_gripper(self, target_pos=None, force=None):
        if target_pos is None:
            target_pos = self.config.gripper.default_close_position
        if force is None:
            force = self.config.gripper.close_force
        self.set_gripper(target_pos, force)
        self.logger.info("夹爪闭合 target_pos=%.4f force=%.2f", target_pos, force)

    def estimate_gripper_force(self, object_id=None):
        if object_id is None:
            object_id = self.object_id

        mass = 0.1
        try:
            mass = p.getDynamicsInfo(object_id, -1)[0]
        except Exception:
            self.logger.exception("读取物体动力学参数失败，使用默认质量")

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
            "mass": mass,
            "object_size": object_size,
            "target_torque": float(target_torque),
            "max_motor_force": float(max_motor_force),
            "hold_force": float(hold_force),
            "finger_target_pos": float(finger_target_pos),
        }

    def get_contact_state(self, object_id=None):
        if object_id is None:
            object_id = self.object_id

        contact_count = 0
        total_normal_force = 0.0
        active_fingers = 0
        for joint_id in self.gripper_joints:
            contacts = p.getContactPoints(bodyA=self.panda_id, bodyB=object_id, linkIndexA=joint_id)
            if contacts:
                active_fingers += 1
            contact_count += len(contacts)
            total_normal_force += sum(contact[9] for contact in contacts)

        return {
            "contact_count": contact_count,
            "active_fingers": active_fingers,
            "total_normal_force": float(total_normal_force),
        }

    def adjust_grip_force(self, object_id=None, base_force=None, max_force=None, target_pos=None):
        if object_id is None:
            object_id = self.object_id
        if base_force is None:
            base_force = self.current_grip_force if self.current_grip_force > 0 else 40.0
        if max_force is None:
            max_force = max(base_force, self.current_grip_force)
        if target_pos is None:
            target_pos = self.current_grip_target

        contact_state = self.get_contact_state(object_id)
        total_normal_force = contact_state["total_normal_force"]
        active_fingers = contact_state["active_fingers"]
        new_force = self.current_grip_force if self.current_grip_force > 0 else base_force

        if active_fingers < 2 or total_normal_force < self.config.gripper.contact_force_threshold:
            new_force = min(max_force, new_force + self.config.gripper.boost_step)
        elif total_normal_force > self.config.gripper.release_contact_force:
            new_force = max(base_force, new_force - self.config.gripper.relax_step)

        self.set_gripper(target_pos, new_force)
        return new_force, contact_state

    def close_with_force_control(self, target_force=None, max_steps=None, object_id=None):
        if object_id is None:
            object_id = self.object_id
        if max_steps is None:
            max_steps = self.config.motion.close_steps

        grasp_cfg = self.estimate_gripper_force(object_id)
        if target_force is None:
            target_force = grasp_cfg["target_torque"]
        target_force = max(0.0, float(target_force))

        max_motor_force = max(
            self.config.gripper.motor_force_min,
            grasp_cfg["max_motor_force"] * self.config.gripper.force_scale,
        )
        hold_force = max(
            self.config.gripper.hold_force_min,
            grasp_cfg["hold_force"] * self.config.gripper.force_scale,
        )
        hold_force = float(np.clip(max(hold_force, target_force * self.config.gripper.force_scale), 12.0, max_motor_force))
        min_finger_pos = grasp_cfg["finger_target_pos"]

        self.logger.info(
            "开始自适应闭合 mass=%.3f size=%s target_torque=%.4f hold_force=%.2f max_force=%.2f",
            grasp_cfg["mass"],
            np.round(grasp_cfg["object_size"], 3),
            target_force,
            hold_force,
            max_motor_force,
        )

        self.open_gripper()
        self.runtime.step(60)

        open_pos = self.config.gripper.open_position
        close_steps = max(20, min(max_steps, self.config.motion.close_steps))
        final_pos = min_finger_pos
        contact_state = {"contact_count": 0, "active_fingers": 0, "total_normal_force": 0.0}

        for step in range(close_steps):
            ratio = (step + 1) / close_steps
            target_pos_now = open_pos + (min_finger_pos - open_pos) * ratio
            motor_force_now = hold_force + (max_motor_force - hold_force) * ratio
            self.set_gripper(target_pos_now, motor_force_now)
            self.runtime.step(4)

            contact_state = self.get_contact_state(object_id)
            if contact_state["active_fingers"] >= 2 and contact_state["total_normal_force"] > self.config.gripper.contact_force_threshold:
                joint_states = [p.getJointState(self.panda_id, joint_id)[0] for joint_id in self.gripper_joints]
                final_pos = max(min_finger_pos, float(np.mean(joint_states)))
                break

        self.set_gripper(final_pos, hold_force)
        for _ in range(self.config.motion.close_stabilize_steps):
            _, contact_state = self.adjust_grip_force(
                object_id=object_id,
                base_force=hold_force,
                max_force=max_motor_force,
                target_pos=final_pos,
            )
            self.runtime.step(1)

        self.logger.info(
            "夹持稳定 final_pos=%.4f current_force=%.2f contact_force=%.4f contacts=%s",
            final_pos,
            self.current_grip_force,
            contact_state["total_normal_force"],
            contact_state["contact_count"],
        )
        return {
            "final_pos": final_pos,
            "hold_force": hold_force,
            "max_motor_force": max_motor_force,
        }
