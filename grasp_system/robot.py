import numpy as np
import pybullet as p

from .config import (
    JOINT_RANGES,
    LOWER_LIMITS,
    PANDA_END_EFFECTOR_INDEX,
    PANDA_NUM_DOFS,
    REST_POSES,
    UPPER_LIMITS,
)


class ArmController:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.arm_joints = list(range(PANDA_NUM_DOFS))

    def get_current_joint_positions(self):
        return [p.getJointState(self.runtime.panda_id, joint_id)[0] for joint_id in self.arm_joints]

    def get_end_effector_position(self):
        link_state = p.getLinkState(self.runtime.panda_id, PANDA_END_EFFECTOR_INDEX, computeForwardKinematics=1)
        return np.array(link_state[4], dtype=float)

    def solve_ik(self, target_pos, target_orn):
        return p.calculateInverseKinematics(
            self.runtime.panda_id,
            PANDA_END_EFFECTOR_INDEX,
            target_pos,
            target_orn,
            lowerLimits=LOWER_LIMITS,
            upperLimits=UPPER_LIMITS,
            jointRanges=JOINT_RANGES,
            restPoses=REST_POSES,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )

    def _set_rendering(self, enabled):
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enabled else 0)
        except Exception:
            pass

    def move_to_joint_positions(self, joint_positions, steps=None, label='arm_joint_move', max_velocity=None, position_gain=None):
        if steps is None:
            steps = self.config.motion.approach_steps
        if max_velocity is None:
            max_velocity = self.config.motion.arm_max_velocity
        if position_gain is None:
            position_gain = self.config.motion.arm_position_gain
        for joint_id in self.arm_joints:
            p.setJointMotorControl2(
                bodyIndex=self.runtime.panda_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_id],
                force=self.config.motion.arm_force,
                maxVelocity=max_velocity,
                positionGain=position_gain,
                velocityGain=self.config.motion.arm_velocity_gain,
            )
        self.logger.info('%s steps=%s', label, steps)
        self.runtime.step(steps)

    def move_home(self):
        self.move_to_joint_positions(REST_POSES, steps=self.config.motion.home_steps, label='move_home')

    def evaluate_pose(self, target_pos, target_orn, joint_poses=None):
        if joint_poses is None:
            joint_poses = self.solve_ik(target_pos, target_orn)

        current_joint_positions = self.get_current_joint_positions()
        try:
            for index, joint_id in enumerate(self.arm_joints):
                p.resetJointState(self.runtime.panda_id, joint_id, joint_poses[index])
            achieved_pos = self.get_end_effector_position()
        finally:
            for index, joint_id in enumerate(self.arm_joints):
                p.resetJointState(self.runtime.panda_id, joint_id, current_joint_positions[index])

        target_pos = np.array(target_pos, dtype=float)
        pos_error = float(np.linalg.norm(achieved_pos - target_pos))

        margin = self.config.motion.joint_limit_margin
        limit_penalty = 0.0
        for index, joint_id in enumerate(self.arm_joints):
            distance_to_limit = min(joint_poses[index] - LOWER_LIMITS[joint_id], UPPER_LIMITS[joint_id] - joint_poses[index])
            if distance_to_limit < margin:
                limit_penalty += (margin - distance_to_limit) / margin

        motion_penalty = float(np.linalg.norm(np.array(joint_poses[: len(self.arm_joints)]) - np.array(current_joint_positions)))
        return {
            'joint_poses': joint_poses,
            'achieved_pos': achieved_pos,
            'pos_error': pos_error,
            'limit_penalty': float(limit_penalty),
            'motion_penalty': motion_penalty,
        }

    def generate_xy_offsets(self):
        step = self.config.motion.grasp_xy_search_step
        radius = self.config.motion.grasp_xy_search_radius
        offsets = [(0.0, 0.0)]
        if step <= 0 or radius <= 0:
            return offsets

        candidates = [
            (step, 0.0),
            (-step, 0.0),
            (0.0, step),
            (0.0, -step),
            (step, step),
            (step, -step),
            (-step, step),
            (-step, -step),
            (radius, 0.0),
            (-radius, 0.0),
            (0.0, radius),
            (0.0, -radius),
        ]
        for dx, dy in candidates:
            if np.hypot(dx, dy) <= radius + 1e-9:
                offsets.append((dx, dy))
        return offsets

    def iter_top_down_pose_candidates(self, target_pos, z_offset=0.0, xy_offsets=None, yaw_candidates=None):
        base_pos = np.array(target_pos, dtype=float)
        if xy_offsets is None:
            xy_offsets = self.generate_xy_offsets()
        if yaw_candidates is None:
            yaw_candidates = self.config.motion.candidate_yaws

        candidates = []
        self._set_rendering(False)
        try:
            for dx, dy in xy_offsets:
                candidate_pos = np.array([base_pos[0] + dx, base_pos[1] + dy, base_pos[2] + z_offset], dtype=float)
                offset_penalty = float(np.hypot(dx, dy) * 10.0)

                for yaw in yaw_candidates:
                    target_orn = p.getQuaternionFromEuler((np.pi, 0.0, float(yaw)))
                    metrics = self.evaluate_pose(candidate_pos, target_orn)
                    score = (
                        metrics['pos_error'] * 150.0
                        + metrics['limit_penalty'] * 2.0
                        + metrics['motion_penalty'] * 0.05
                        + offset_penalty
                    )
                    candidates.append({
                        'target_pos': candidate_pos,
                        'target_orn': target_orn,
                        'yaw': float(yaw),
                        'score': float(score),
                        **metrics,
                    })
        finally:
            self._set_rendering(True)

        candidates.sort(key=lambda item: item['score'])
        return candidates

    def move_to_position(self, target_pos, target_orn=None, steps=None, grip_adjust=None, joint_poses=None, label='arm_move', max_velocity=None, position_gain=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler(self.config.motion.default_orientation_euler)
        if steps is None:
            steps = self.config.motion.approach_steps
        if joint_poses is None:
            joint_poses = self.solve_ik(target_pos, target_orn)
        if max_velocity is None:
            max_velocity = self.config.motion.arm_max_velocity
        if position_gain is None:
            position_gain = self.config.motion.arm_position_gain

        for joint_id in self.arm_joints:
            p.setJointMotorControl2(
                bodyIndex=self.runtime.panda_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
                force=self.config.motion.arm_force,
                maxVelocity=max_velocity,
                positionGain=position_gain,
                velocityGain=self.config.motion.arm_velocity_gain,
            )

        self.logger.info('%s target_pos=%s steps=%s', label, np.round(np.array(target_pos, dtype=float), 4).tolist(), steps)
        self.runtime.step(steps, hook=grip_adjust)

    def verify_reached(self, target_pos, tolerance=None):
        if tolerance is None:
            tolerance = self.config.motion.ik_position_tolerance
        current_pos = self.get_end_effector_position()
        target_pos = np.array(target_pos, dtype=float)
        error = float(np.linalg.norm(current_pos - target_pos))
        return error <= tolerance, current_pos, error

    def move_and_verify(self, target_pos, target_orn=None, steps=None, grip_adjust=None, joint_poses=None, label='arm_move', max_velocity=None, position_gain=None):
        self.move_to_position(
            target_pos,
            target_orn=target_orn,
            steps=steps,
            grip_adjust=grip_adjust,
            joint_poses=joint_poses,
            label=label,
            max_velocity=max_velocity,
            position_gain=position_gain,
        )

        reached, current_pos, error = self.verify_reached(target_pos)
        extra_steps = 0
        while not reached and extra_steps < self.config.motion.reach_check_max_extra_steps:
            interval = self.config.motion.reach_check_interval
            self.runtime.step(interval, hook=grip_adjust)
            extra_steps += interval
            reached, current_pos, error = self.verify_reached(target_pos)

        self.logger.info(
            'MOVE_VERIFY label=%s reached=%s error=%.4f extra_steps=%d current=%s target=%s',
            label,
            reached,
            error,
            extra_steps,
            np.round(current_pos, 4).tolist(),
            np.round(np.array(target_pos, dtype=float), 4).tolist(),
        )
        return reached, current_pos, error
