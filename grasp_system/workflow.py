from typing import Optional

import numpy as np
import pybullet as p

from .config import AppConfig, PANDA_END_EFFECTOR_INDEX
from .gripper import GripperController
from .logging_utils import setup_logger
from .robot import ArmController
from .simulation import SimulationRuntime
from .vision import VisionSystem


class GraspWorkflow:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.logger = setup_logger(
            self.config.logging.logger_name,
            self.config.logging.level,
            self.config.logging.log_file,
        )
        self.runtime = SimulationRuntime(self.config, self.logger)
        self.vision = VisionSystem(self.runtime, self.config, self.logger)
        self.gripper = GripperController(self.runtime, self.config, self.logger)
        self.arm = ArmController(self.runtime, self.config, self.logger)

    @staticmethod
    def _fmt_vec(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    def _get_true_object_position(self):
        true_pos, _ = p.getBasePositionAndOrientation(self.runtime.obj_id)
        return np.array(true_pos, dtype=float)

    def _get_end_effector_position(self):
        link_state = p.getLinkState(self.runtime.panda_id, PANDA_END_EFFECTOR_INDEX, computeForwardKinematics=1)
        return np.array(link_state[4], dtype=float)

    def _log_end_effector_state(self, stage, target_pos=None):
        ee_pos = self._get_end_effector_position()
        obj_pos = self._get_true_object_position()
        parts = [
            f'stage={stage}',
            f'ee_pos={self._fmt_vec(ee_pos)}',
            f'api_true={self._fmt_vec(obj_pos)}',
            f'ee_to_object={self._fmt_vec(ee_pos - obj_pos)}',
        ]
        if target_pos is not None:
            target_pos = np.array(target_pos, dtype=float)
            parts.append(f'target={self._fmt_vec(target_pos)}')
            parts.append(f'ee_to_target={self._fmt_vec(ee_pos - target_pos)}')
        self.logger.info('EE_STATE %s', ' | '.join(parts))

    def _log_target_compare(self, stage, vision_pos, planned_pos=None):
        true_pos = self._get_true_object_position()
        message = [
            f'{stage} vision={self._fmt_vec(vision_pos)}',
            f'api_true={self._fmt_vec(true_pos)}',
            f'vision_delta={self._fmt_vec(np.array(vision_pos, dtype=float) - true_pos)}',
        ]
        if planned_pos is not None:
            planned_pos = np.array(planned_pos, dtype=float)
            message.append(f'planned={self._fmt_vec(planned_pos)}')
            message.append(f'plan_delta={self._fmt_vec(planned_pos - true_pos)}')
        self.logger.info('TARGET_COMPARE %s', ' | '.join(message))

    def _maybe_refine_object_position(self, current_pos):
        refined_pos = self.vision.locate_object(self.runtime.obj_id)
        delta = np.linalg.norm(np.array(refined_pos) - np.array(current_pos))
        replan_threshold = self.config.vision.replan_position_threshold
        accept_threshold = self.config.vision.refine_accept_max_shift

        if delta > accept_threshold:
            self.logger.warning(
                'REFINE_REJECT delta=%.4f accept_threshold=%.4f previous=%s refined=%s',
                delta,
                accept_threshold,
                self._fmt_vec(current_pos),
                self._fmt_vec(refined_pos),
            )
            return np.array(current_pos)

        if delta >= replan_threshold:
            fused_pos = 0.5 * (np.array(current_pos) + np.array(refined_pos))
            self.logger.info(
                'REFINE_ACCEPT delta=%.4f previous=%s refined=%s fused=%s',
                delta,
                self._fmt_vec(current_pos),
                self._fmt_vec(refined_pos),
                self._fmt_vec(fused_pos),
            )
            return fused_pos

        self.logger.info(
            'REFINE_KEEP delta=%.4f previous=%s refined=%s',
            delta,
            self._fmt_vec(current_pos),
            self._fmt_vec(refined_pos),
        )
        return np.array(refined_pos)

    def _check_lift_success(self, expected_ee_pos, object_z_before):
        ee_pos = self._get_end_effector_position()
        obj_pos = self._get_true_object_position()
        obj_height_gain = float(obj_pos[2] - object_z_before)
        obj_xy_error = float(np.linalg.norm(obj_pos[:2] - ee_pos[:2]))
        success = (
            obj_height_gain >= self.config.motion.lift_success_min_height_delta
            and obj_xy_error <= self.config.motion.lift_success_max_xy_error
        )
        self.logger.info(
            'LIFT_CHECK success=%s obj_height_gain=%.4f obj_xy_error=%.4f ee_pos=%s obj_pos=%s expected_ee=%s',
            success,
            obj_height_gain,
            obj_xy_error,
            self._fmt_vec(ee_pos),
            self._fmt_vec(obj_pos),
            self._fmt_vec(expected_ee_pos),
        )
        return success

    def _execute_grasp_attempt(self, obj_pos, plan, attempt_index):
        if attempt_index > 1:
            self.logger.info('GRASP_ATTEMPT_RESET index=%d', attempt_index)
            self.arm.move_home()

        pre_grasp_pos = np.array(plan['target_pos'], dtype=float)
        pre_grasp_pos[2] = obj_pos[2] + self.config.motion.pre_grasp_offset

        self.logger.info(
            'GRASP_ATTEMPT index=%d yaw=%.3f score=%.4f grasp_pos=%s pre_grasp=%s',
            attempt_index,
            plan['yaw'],
            plan['score'],
            self._fmt_vec(plan['target_pos']),
            self._fmt_vec(pre_grasp_pos),
        )

        self.gripper.open_gripper()
        pre_ok, _, _ = self.arm.move_and_verify(
            pre_grasp_pos,
            target_orn=plan['target_orn'],
            label=f'move_pre_grasp_attempt_{attempt_index}',
        )
        if not pre_ok:
            self.logger.warning('GRASP_ATTEMPT_PRE_FAIL index=%d yaw=%.3f', attempt_index, plan['yaw'])
            return False, None

        descend_ok, _, _ = self.arm.move_and_verify(
            plan['target_pos'],
            target_orn=plan['target_orn'],
            joint_poses=plan['joint_poses'],
            label=f'move_grasp_attempt_{attempt_index}',
        )
        self._log_end_effector_state(f'after_descend_attempt_{attempt_index}', plan['target_pos'])
        if not descend_ok:
            self.logger.warning('GRASP_ATTEMPT_DESCEND_FAIL index=%d yaw=%.3f', attempt_index, plan['yaw'])
            return False, None

        if self.config.logging.log_contact_details:
            self.gripper.log_contact_state(f'before_close_attempt_{attempt_index}')
        grip_state = self.gripper.close_with_fixed_force()

        object_pos_before_lift = self._get_true_object_position()
        validation_lift_pos = np.array(plan['target_pos'], dtype=float)
        validation_lift_pos[2] += self.config.motion.validation_lift_offset
        lift_ok, _, _ = self.arm.move_and_verify(
            validation_lift_pos,
            target_orn=plan['target_orn'],
            label=f'move_validation_lift_attempt_{attempt_index}',
            steps=self.config.motion.lift_steps,
            max_velocity=self.config.motion.lift_arm_max_velocity,
            position_gain=self.config.motion.lift_arm_position_gain,
        )
        stable = lift_ok and self._check_lift_success(validation_lift_pos, object_pos_before_lift[2])
        self._log_end_effector_state(f'after_validation_lift_attempt_{attempt_index}', validation_lift_pos)

        if not stable:
            self.logger.warning('GRASP_ATTEMPT_LIFT_FAIL index=%d yaw=%.3f', attempt_index, plan['yaw'])
            self.gripper.open_gripper()
            self.runtime.step(30)
            return False, grip_state

        self.logger.info(
            'GRASP_ATTEMPT_RESULT index=%d success=True hold_force=%.2f contact_count=%d active_fingers=%d',
            attempt_index,
            grip_state['hold_force'],
            grip_state['contact_state']['contact_count'],
            grip_state['contact_state']['active_fingers'],
        )
        return True, {'grip_state': grip_state, 'current_lift_pos': validation_lift_pos}

    def execute_place(self):
        release_position = np.array(self.config.place.release_position, dtype=float)
        place_plan = self.arm.plan_top_down_pose(
            release_position,
            z_offset=0.0,
            xy_offsets=[(0.0, 0.0)],
            label='place',
        )
        pre_place_pos = np.array(place_plan['target_pos'], dtype=float)
        pre_place_pos[2] += self.config.place.pre_place_offset
        retreat_pos = np.array(place_plan['target_pos'], dtype=float)
        retreat_pos[2] += self.config.place.retreat_offset

        self.arm.move_to_position(pre_place_pos, target_orn=place_plan['target_orn'], label='move_pre_place')
        self.arm.move_to_position(
            place_plan['target_pos'],
            target_orn=place_plan['target_orn'],
            joint_poses=place_plan['joint_poses'],
            label='move_release',
        )
        self.gripper.open_gripper()
        self.runtime.step(self.config.place.settle_steps)
        self.arm.move_to_position(retreat_pos, target_orn=place_plan['target_orn'], label='move_post_place')

    def run(self, gui: bool = True):
        self.runtime.connect(gui=gui)
        try:
            self.logger.info('Start grasp workflow')
            self.runtime.step(self.config.motion.settle_steps)
            self.arm.move_home()

            initial_obj_pos = self.vision.locate_object(self.runtime.obj_id)
            refined_obj_pos = self._maybe_refine_object_position(initial_obj_pos)

            candidates = self.arm.iter_top_down_pose_candidates(
                refined_obj_pos,
                z_offset=self.config.motion.grasp_offset,
                xy_offsets=[(0.0, 0.0)],
            )
            if not candidates:
                raise RuntimeError('No grasp pose candidates available')

            success = False
            chosen_plan = None
            attempt_payload = None

            for attempt_index, plan in enumerate(candidates[:3], start=1):
                self._log_target_compare(f'grasp_attempt_{attempt_index}', refined_obj_pos, plan['target_pos'])
                success, attempt_payload = self._execute_grasp_attempt(refined_obj_pos, plan, attempt_index)
                if success:
                    chosen_plan = plan
                    break

            if not success or chosen_plan is None or attempt_payload is None:
                self.logger.error('GRASP_FAILED no candidate achieved a stable lifted grasp')
                return

            self.logger.info('GRASP_SUCCESS yaw=%.3f target=%s', chosen_plan['yaw'], self._fmt_vec(chosen_plan['target_pos']))
            self.runtime.step(self.config.motion.hold_steps)

            lift_pos = np.array(chosen_plan['target_pos'], dtype=float)
            lift_pos[2] += self.config.motion.lift_offset
            self.logger.info('Lift target %s', self._fmt_vec(lift_pos))
            self.arm.move_and_verify(
                lift_pos,
                target_orn=chosen_plan['target_orn'],
                label='move_lift',
                steps=self.config.motion.lift_steps,
                max_velocity=self.config.motion.lift_arm_max_velocity,
                position_gain=self.config.motion.lift_arm_position_gain,
            )
            self._log_end_effector_state('after_lift', lift_pos)

            if self.config.place.enabled:
                self.execute_place()

            self.logger.info('Workflow finished')
        finally:
            self.runtime.disconnect()
