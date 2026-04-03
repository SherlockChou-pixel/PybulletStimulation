from typing import Any, Optional

import numpy as np
import pybullet as p

from .config import AppConfig, PANDA_END_EFFECTOR_INDEX
from .gripper import GripperController
from .imu import IMUSystem
from .logging_utils import setup_logger
from .multimodal import MultiModalObserver
from .pressure import PressureSystem
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
        self.imu = IMUSystem(self.runtime, self.config, self.logger)
        self.pressure = PressureSystem(self.runtime, self.config, self.logger)
        self.observer = MultiModalObserver(self.runtime, self.imu, self.pressure, self.config, self.logger)
        self.latest_vision_position = None
        self._last_fusion_control_step = -10**9
        self._fusion_stage_actions = {}
        self._reset_run_metrics()

    @staticmethod
    def _fmt_vec(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    def _get_true_object_position(self):
        true_pos, _ = p.getBasePositionAndOrientation(self.runtime.obj_id)
        return np.array(true_pos, dtype=float)

    def _get_end_effector_position(self):
        link_state = p.getLinkState(self.runtime.panda_id, PANDA_END_EFFECTOR_INDEX, computeForwardKinematics=1)
        return np.array(link_state[4], dtype=float)

    def _reset_run_metrics(self):
        self._run_stage_records: list[tuple[str, dict[str, Any]]] = []
        self._run_attempts: list[dict[str, Any]] = []
        self._fusion_control_count = 0
        self._recenter_count = 0
        self._place_attempted = False
        self._place_success = False
        self._failure_reason: Optional[str] = None

    def _record_observation(self, stage, observation):
        if observation is not None:
            self._run_stage_records.append((stage, observation))
        return observation

    def _observe_stage(self, stage, vision_position=None, planned_position=None, object_position=None):
        observation = self.observer.log_stage(
            stage,
            vision_position=vision_position,
            planned_position=planned_position,
            object_position=object_position,
        )
        return self._record_observation(stage, observation)

    def _find_stage_observation(self, prefix):
        for stage, observation in reversed(self._run_stage_records):
            if stage == prefix or stage.startswith(prefix):
                return observation
        return None

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fusion_value(self, observation, key):
        fusion = (observation or {}).get('fusion') or {}
        return self._safe_float(fusion.get(key))

    def _fusion_state(self, observation):
        fusion = (observation or {}).get('fusion') or {}
        return fusion.get('state')

    def _set_failure_reason(self, reason):
        if not self._failure_reason:
            self._failure_reason = str(reason)

    def _build_run_result(self, status, chosen_plan=None, attempt_payload=None):
        after_vision = self._find_stage_observation('after_vision_lock')
        after_close = self._find_stage_observation('after_close_attempt_')
        after_validation = self._find_stage_observation('after_validation_lift_attempt_')
        after_lift = self._find_stage_observation('after_lift')
        before_release = self._find_stage_observation('before_release')
        after_release = self._find_stage_observation('after_release')

        vision_fusion = (after_vision or {}).get('fusion') or {}
        vision_error = vision_fusion.get('vision_error')

        result = {
            'scenario_name': self.config.scene.scenario_name,
            'status': status,
            'workflow_finished': status == 'success',
            'grasp_success': chosen_plan is not None and attempt_payload is not None,
            'place_attempted': self._place_attempted,
            'place_success': self._place_success,
            'failure_reason': None if status == 'success' else self._failure_reason,
            'sim_time': float(self.runtime.sim_time),
            'sim_steps': int(self.runtime.sim_step_count),
            'object_spawn_x': None,
            'object_spawn_y': None,
            'object_spawn_z': None,
            'object_spawn_roll': None,
            'object_spawn_pitch': None,
            'object_spawn_yaw': None,
            'vision_error': self._safe_float(vision_error),
            'vision_confidence': self._safe_float(vision_fusion.get('vision_confidence')),
            'candidate_count': len(self._run_attempts),
            'attempts_used': len(self._run_attempts),
            'had_retry': len(self._run_attempts) > 1,
            'fusion_control_count': int(self._fusion_control_count),
            'recenter_count': int(self._recenter_count),
            'chosen_yaw': None,
            'chosen_roll_deg': None,
            'chosen_pitch_deg': None,
            'chosen_score': None,
            'initial_hold_force': None,
            'final_grip_force': self._safe_float(self.gripper.current_grip_force),
            'after_close_grasp_confidence': self._fusion_value(after_close, 'grasp_confidence'),
            'after_close_slip_risk': self._fusion_value(after_close, 'slip_risk'),
            'after_close_pressure_confidence': self._fusion_value(after_close, 'pressure_confidence'),
            'after_validation_grasp_confidence': self._fusion_value(after_validation, 'grasp_confidence'),
            'after_validation_slip_risk': self._fusion_value(after_validation, 'slip_risk'),
            'after_validation_pressure_confidence': self._fusion_value(after_validation, 'pressure_confidence'),
            'after_lift_state': self._fusion_state(after_lift),
            'after_lift_grasp_confidence': self._fusion_value(after_lift, 'grasp_confidence'),
            'after_lift_slip_risk': self._fusion_value(after_lift, 'slip_risk'),
            'before_release_state': self._fusion_state(before_release),
            'before_release_grasp_confidence': self._fusion_value(before_release, 'grasp_confidence'),
            'before_release_slip_risk': self._fusion_value(before_release, 'slip_risk'),
            'after_release_state': self._fusion_state(after_release),
        }

        if self.runtime.object_spawn_position is not None:
            spawn = np.array(self.runtime.object_spawn_position, dtype=float)
            result.update(
                {
                    'object_spawn_x': float(spawn[0]),
                    'object_spawn_y': float(spawn[1]),
                    'object_spawn_z': float(spawn[2]),
                }
            )

        if self.runtime.object_spawn_orientation is not None:
            roll, pitch, yaw = p.getEulerFromQuaternion(np.array(self.runtime.object_spawn_orientation, dtype=float).tolist())
            result.update(
                {
                    'object_spawn_roll': float(roll),
                    'object_spawn_pitch': float(pitch),
                    'object_spawn_yaw': float(yaw),
                }
            )

        if chosen_plan is not None:
            result.update(
                {
                    'chosen_yaw': self._safe_float(chosen_plan.get('yaw')),
                    'chosen_roll_deg': self._safe_float(chosen_plan.get('roll_deg')),
                    'chosen_pitch_deg': self._safe_float(chosen_plan.get('pitch_deg')),
                    'chosen_score': self._safe_float(chosen_plan.get('score')),
                }
            )

        if attempt_payload is not None:
            grip_state = attempt_payload.get('grip_state') or {}
            result['initial_hold_force'] = self._safe_float(grip_state.get('hold_force'))

        return result

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
        if not self.config.vision.enable_refine_pass:
            self.logger.info('REFINE_SKIP enable_refine_pass=False position=%s', self._fmt_vec(current_pos))
            return np.array(current_pos)

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

    def _activate_multimodal_systems(self):
        self.imu.activate()
        self.pressure.activate()

    def _deactivate_multimodal_systems(self):
        self.imu.deactivate()
        self.pressure.deactivate()

    def _log_fusion_warning(self, observation):
        fusion = (observation or {}).get('fusion') or {}
        if not fusion:
            return

        if fusion.get('slip_risk', 0.0) >= self.config.fusion.high_slip_risk:
            self.logger.warning(
                'FUSION_ALERT state=%s slip_risk=%.3f grasp_confidence=%.3f alerts=%s',
                fusion.get('state'),
                fusion.get('slip_risk', 0.0),
                fusion.get('grasp_confidence', 0.0),
                fusion.get('alerts', []),
            )
        elif fusion.get('alerts'):
            self.logger.info(
                'FUSION_HINT state=%s grasp_confidence=%.3f alerts=%s',
                fusion.get('state'),
                fusion.get('grasp_confidence', 0.0),
                fusion.get('alerts', []),
            )

    def _is_grasp_quality_acceptable(self, observation):
        fusion = (observation or {}).get('fusion') or {}
        if not fusion:
            return True, []

        reasons = []
        if fusion.get('grasp_confidence', 0.0) < self.config.fusion.validation_min_grasp_confidence:
            reasons.append(f'grasp_confidence={fusion.get("grasp_confidence", 0.0):.3f}')
        if fusion.get('slip_risk', 0.0) > self.config.fusion.validation_max_slip_risk:
            reasons.append(f'slip_risk={fusion.get("slip_risk", 0.0):.3f}')
        if fusion.get('pressure_lateral_ratio', 0.0) > self.config.fusion.validation_max_lateral_ratio:
            reasons.append(f'lateral_ratio={fusion.get("pressure_lateral_ratio", 0.0):.3f}')
        return len(reasons) == 0, reasons

    @staticmethod
    def _is_transport_stage_name(stage):
        return 'transport_monitor' in stage

    @staticmethod
    def _is_place_transport_stage(stage):
        return stage in {'transport_monitor_pre_place', 'transport_monitor_release'}

    def _maybe_recenter_grasp(self, stage, target_orn, anchor_pos, attempt_index):
        if not self.config.fusion.recenter_enabled:
            return np.array(anchor_pos, dtype=float), None

        latest_observation = None
        current_anchor = np.array(anchor_pos, dtype=float)
        max_attempts = max(0, int(self.config.fusion.recenter_max_attempts))

        for recenter_index in range(1, max_attempts + 1):
            ee_pos = self._get_end_effector_position()
            obj_pos = self._get_true_object_position()
            probe = self.observer.capture(
                f'{stage}_probe_{recenter_index}',
                planned_position=current_anchor,
                object_position=obj_pos,
            )
            fusion = probe.get('fusion') or {}
            lateral_ratio = float(fusion.get('pressure_lateral_ratio', 0.0))
            xy_delta = np.array(obj_pos[:2] - ee_pos[:2], dtype=float)
            xy_delta_norm = float(np.linalg.norm(xy_delta))
            needs_recenter = (
                lateral_ratio >= self.config.fusion.recenter_lateral_ratio_trigger
                or xy_delta_norm >= self.config.fusion.recenter_object_xy_trigger
            )
            if not needs_recenter:
                latest_observation = probe
                break

            correction_xy = xy_delta * float(self.config.fusion.recenter_xy_gain)
            correction_xy = np.clip(
                correction_xy,
                -float(self.config.fusion.recenter_max_step),
                float(self.config.fusion.recenter_max_step),
            )
            if np.linalg.norm(correction_xy) < 1e-4:
                latest_observation = probe
                break

            recenter_target = ee_pos.copy()
            recenter_target[:2] += correction_xy
            self._recenter_count += 1
            self.logger.warning(
                'GRASP_RECENTER stage=%s attempt=%d recenter_index=%d ee_pos=%s object_pos=%s xy_delta=%s correction=%s lateral_ratio=%.3f',
                stage,
                attempt_index,
                recenter_index,
                self._fmt_vec(ee_pos),
                self._fmt_vec(obj_pos),
                self._fmt_vec(np.append(xy_delta, 0.0)),
                self._fmt_vec(np.append(correction_xy, 0.0)),
                lateral_ratio,
            )
            self.arm.move_and_verify(
                recenter_target,
                target_orn=target_orn,
                label=f'move_recenter_attempt_{attempt_index}_{recenter_index}',
                steps=self.config.motion.recenter_steps,
                max_velocity=self.config.motion.lift_arm_max_velocity,
                position_gain=self.config.motion.lift_arm_position_gain,
                grip_adjust=self._build_fusion_control_hook(
                    f'transport_monitor_recenter_attempt_{attempt_index}_{recenter_index}',
                    planned_position=recenter_target,
                ),
            )
            self.runtime.step(self.config.motion.recenter_settle_steps)
            current_anchor = recenter_target
            self._log_end_effector_state(
                f'after_recenter_attempt_{attempt_index}_{recenter_index}',
                recenter_target,
            )
            latest_observation = self._observe_stage(
                f'after_recenter_attempt_{attempt_index}_{recenter_index}',
                planned_position=recenter_target,
                object_position=self._get_true_object_position(),
            )
            self._log_fusion_warning(latest_observation)
            lift_stage_state = self._fusion_stage_actions.setdefault('transport_monitor_lift', {'count': 0})
            lift_stage_state['count'] = 0
            lift_stage_state['pause_until'] = int(self.runtime.sim_step_count) + int(self.config.fusion.post_recenter_grace_steps)
            self._last_fusion_control_step = int(self.runtime.sim_step_count)

            acceptable, _ = self._is_grasp_quality_acceptable(latest_observation)
            if acceptable:
                break

        return current_anchor, latest_observation

    def _build_fusion_control_hook(self, stage, planned_position=None):
        def hook():
            self.gripper.maintain_grip()

            if not self.config.fusion.closed_loop_enabled:
                return

            interval = max(1, int(self.config.fusion.control_monitor_interval))
            current_step = int(self.runtime.sim_step_count)
            if current_step % interval != 0:
                return

            observation = self.observer.capture(
                stage,
                vision_position=None,
                planned_position=planned_position,
                object_position=self._get_true_object_position(),
            )
            fusion = observation.get('fusion') or {}
            if not fusion:
                return

            should_tighten = False
            reasons = []
            fusion_state = fusion.get('state')
            if fusion_state in {'SLIP_RISK', 'GRASP_LOST', 'TRANSPORT_UNCERTAIN', 'GRASP_UNCERTAIN'}:
                should_tighten = True
                reasons.append(f'state={fusion_state}')
            if fusion.get('slip_risk', 0.0) >= self.config.fusion.control_slip_risk:
                should_tighten = True
                reasons.append(f'slip={fusion.get("slip_risk", 0.0):.3f}')
            if fusion.get('pressure_confidence', 1.0) <= self.config.fusion.control_low_pressure_confidence:
                should_tighten = True
                reasons.append(f'pressure_conf={fusion.get("pressure_confidence", 0.0):.3f}')
            if (
                fusion.get('grasp_confidence', 1.0) <= self.config.fusion.control_low_grasp_confidence
                and fusion_state not in {'TRANSPORT_STABLE', 'GRASP_STABLE'}
            ):
                should_tighten = True
                reasons.append(f'grasp_conf={fusion.get("grasp_confidence", 0.0):.3f}')

            if not should_tighten:
                return

            stage_state = self._fusion_stage_actions.setdefault(stage, {'count': 0})
            if current_step < int(stage_state.get('pause_until', -10**9)):
                return
            severe_state = fusion_state in {'SLIP_RISK', 'GRASP_LOST'}
            severe_risk = fusion.get('slip_risk', 0.0) >= self.config.fusion.high_slip_risk
            lateral_only_instability = (
                fusion.get('pressure_lateral_ratio', 0.0) >= self.config.fusion.recenter_lateral_ratio_trigger
                and fusion.get('object_speed', 0.0) <= self.config.fusion.control_motion_stable_threshold
                and fusion.get('pressure_normal', 0.0) >= 0.9 * self.config.fusion.desired_total_normal_force
                and self.gripper.current_grip_force >= self.config.fusion.control_force_hold_threshold
            )
            if lateral_only_instability and not (severe_state or severe_risk):
                return

            if self._is_transport_stage_name(stage) and not (severe_state or severe_risk):
                pressure_good = fusion.get('pressure_confidence', 0.0) >= self.config.fusion.control_transport_pressure_confidence
                lateral_good = fusion.get('pressure_lateral_ratio', 0.0) <= self.config.fusion.control_transport_lateral_ratio
                motion_soft = fusion.get('object_speed', 0.0) <= self.config.fusion.control_transport_object_speed
                grasp_soft = fusion.get('grasp_confidence', 0.0) >= self.config.fusion.control_transport_low_grasp_confidence
                slip_soft = fusion.get('slip_risk', 0.0) <= self.config.fusion.control_transport_slip_risk
                if pressure_good and lateral_good and ((slip_soft and motion_soft) or grasp_soft):
                    return

            if self._is_place_transport_stage(stage) and fusion_state == 'TRANSPORT_STABLE' and not (severe_state or severe_risk):
                place_guard = (
                    fusion.get('slip_risk', 0.0) <= self.config.fusion.control_place_stable_slip_risk
                    and fusion.get('grasp_confidence', 0.0) >= self.config.fusion.control_place_stable_grasp_confidence
                    and fusion.get('pressure_confidence', 0.0) >= self.config.fusion.control_place_stable_pressure_confidence
                    and fusion.get('pressure_lateral_ratio', 0.0) <= self.config.fusion.control_place_stable_lateral_ratio
                )
                if place_guard:
                    return
                if stage_state['count'] >= self.config.fusion.control_place_max_actions_stable:
                    return

            if (
                stage_state['count'] >= self.config.fusion.control_max_actions_per_stage
                and not (severe_state or severe_risk)
            ):
                return

            cooldown_steps = max(1, int(self.config.fusion.control_cooldown_steps))
            if current_step - self._last_fusion_control_step < cooldown_steps:
                return

            at_force_cap = self.gripper.current_grip_force >= self.config.fusion.control_force_cap - 1e-6
            at_target_floor = self.gripper.current_grip_target <= self.config.gripper.default_close_position + 1e-6
            if at_force_cap and at_target_floor:
                self._last_fusion_control_step = current_step
                return

            action = self.gripper.adaptive_tighten(reason=f'{stage}|{",".join(reasons)}')
            if not action.get('changed', False):
                self._last_fusion_control_step = current_step
                return
            stage_state['count'] += 1
            self._fusion_control_count += 1
            self._last_fusion_control_step = current_step
            self.logger.warning(
                'FUSION_CONTROL stage=%s state=%s slip_risk=%.3f grasp_confidence=%.3f pressure_confidence=%.3f action_force=%.2f action_target=%.4f action_count=%d alerts=%s',
                stage,
                fusion_state,
                fusion.get('slip_risk', 0.0),
                fusion.get('grasp_confidence', 0.0),
                fusion.get('pressure_confidence', 0.0),
                action['force'],
                action['target_pos'],
                stage_state['count'],
                fusion.get('alerts', []),
            )

        return hook

    def _execute_grasp_attempt(self, obj_pos, plan, attempt_index):
        if attempt_index > 1:
            self.logger.info('GRASP_ATTEMPT_RESET index=%d', attempt_index)
            self.arm.move_home()

        pre_grasp_pos = np.array(plan['target_pos'], dtype=float)
        pre_grasp_pos[2] = obj_pos[2] + self.config.motion.pre_grasp_offset

        self.logger.info(
            'GRASP_ATTEMPT index=%d yaw=%.3f roll_deg=%.1f pitch_deg=%.1f score=%.4f grasp_pos=%s pre_grasp=%s',
            attempt_index,
            plan['yaw'],
            plan.get('roll_deg', 0.0),
            plan.get('pitch_deg', 0.0),
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
            self._set_failure_reason('pre_grasp_not_reached')
            self.logger.warning('GRASP_ATTEMPT_PRE_FAIL index=%d yaw=%.3f', attempt_index, plan['yaw'])
            return False, None

        descend_ok, _, _ = self.arm.move_and_verify(
            plan['target_pos'],
            target_orn=plan['target_orn'],
            joint_poses=plan['joint_poses'],
            label=f'move_grasp_attempt_{attempt_index}',
        )
        self._log_end_effector_state(f'after_descend_attempt_{attempt_index}', plan['target_pos'])
        descend_observation = self._observe_stage(
            f'after_descend_attempt_{attempt_index}',
            planned_position=plan['target_pos'],
            object_position=self._get_true_object_position(),
        )
        self._log_fusion_warning(descend_observation)
        if not descend_ok:
            self._set_failure_reason('descend_not_reached')
            self.logger.warning('GRASP_ATTEMPT_DESCEND_FAIL index=%d yaw=%.3f', attempt_index, plan['yaw'])
            return False, None

        if self.config.logging.log_contact_details:
            self.gripper.log_contact_state(f'before_close_attempt_{attempt_index}')
        grip_state = self.gripper.close_with_fixed_force()
        pressure_summary = self.pressure.build_summary()
        pressure_ready = bool(pressure_summary and pressure_summary['stable_contact'])
        close_observation = self._observe_stage(
            f'after_close_attempt_{attempt_index}',
            planned_position=plan['target_pos'],
            object_position=self._get_true_object_position(),
        )
        self._log_fusion_warning(close_observation)
        if not pressure_ready:
            self.logger.warning(
                'PRESSURE_CHECK_FAIL index=%d active_fingers=%s normal_force=%s',
                attempt_index,
                None if pressure_summary is None else pressure_summary['active_fingers'],
                None if pressure_summary is None else round(pressure_summary['total_normal_force'], 4),
            )

        object_pos_before_lift = self._get_true_object_position()
        validation_lift_pos = np.array(plan['target_pos'], dtype=float)
        validation_lift_pos[2] += self.config.motion.validation_lift_offset
        validation_hook = self._build_fusion_control_hook(
            f'transport_monitor_validation_lift_attempt_{attempt_index}',
            planned_position=validation_lift_pos,
        )
        lift_ok, _, _ = self.arm.move_and_verify(
            validation_lift_pos,
            target_orn=plan['target_orn'],
            label=f'move_validation_lift_attempt_{attempt_index}',
            steps=self.config.motion.lift_steps,
            max_velocity=self.config.motion.lift_arm_max_velocity,
            position_gain=self.config.motion.lift_arm_position_gain,
            grip_adjust=validation_hook,
        )
        stable = pressure_ready and lift_ok and self._check_lift_success(validation_lift_pos, object_pos_before_lift[2])
        self._log_end_effector_state(f'after_validation_lift_attempt_{attempt_index}', validation_lift_pos)
        lift_observation = self._observe_stage(
            f'after_validation_lift_attempt_{attempt_index}',
            planned_position=validation_lift_pos,
            object_position=self._get_true_object_position(),
        )
        self._log_fusion_warning(lift_observation)
        validation_lift_pos, recenter_observation = self._maybe_recenter_grasp(
            'validation_lift',
            plan['target_orn'],
            validation_lift_pos,
            attempt_index,
        )
        if recenter_observation is not None:
            lift_observation = recenter_observation

        quality_ok, quality_reasons = self._is_grasp_quality_acceptable(lift_observation)
        attempt_record = {
            'attempt_index': int(attempt_index),
            'yaw': self._safe_float(plan.get('yaw')),
            'roll_deg': self._safe_float(plan.get('roll_deg')),
            'pitch_deg': self._safe_float(plan.get('pitch_deg')),
            'score': self._safe_float(plan.get('score')),
            'pre_grasp_ok': bool(pre_ok),
            'descend_ok': bool(descend_ok),
            'pressure_ready': bool(pressure_ready),
            'lift_motion_ok': bool(lift_ok),
            'stable': bool(stable),
            'quality_ok': bool(quality_ok),
            'quality_reasons': list(quality_reasons),
        }
        self._run_attempts.append(attempt_record)
        if not quality_ok:
            self.logger.warning(
                'GRASP_QUALITY_REJECT index=%d yaw=%.3f reasons=%s',
                attempt_index,
                plan['yaw'],
                quality_reasons,
            )

        if not stable or not quality_ok:
            if not pressure_ready:
                self._set_failure_reason('pressure_contact_not_stable')
            elif not lift_ok:
                self._set_failure_reason('validation_lift_not_reached')
            elif not stable:
                self._set_failure_reason('lift_validation_failed')
            elif not quality_ok:
                self._set_failure_reason(f'grasp_quality_reject:{";".join(quality_reasons)}')
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
        self._place_attempted = True
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

        self.logger.info('PLACE_START release=%s', self._fmt_vec(place_plan['target_pos']))
        self.arm.move_and_verify(
            pre_place_pos,
            target_orn=place_plan['target_orn'],
            label='move_pre_place',
            grip_adjust=self._build_fusion_control_hook('transport_monitor_pre_place', planned_position=pre_place_pos),
            max_velocity=self.config.motion.lift_arm_max_velocity,
            position_gain=self.config.motion.lift_arm_position_gain,
        )
        self.arm.move_and_verify(
            place_plan['target_pos'],
            target_orn=place_plan['target_orn'],
            joint_poses=place_plan['joint_poses'],
            label='move_release',
            grip_adjust=self._build_fusion_control_hook('transport_monitor_release', planned_position=place_plan['target_pos']),
            max_velocity=self.config.motion.lift_arm_max_velocity,
            position_gain=self.config.motion.lift_arm_position_gain,
        )
        before_release_observation = self._observe_stage(
            'before_release',
            planned_position=place_plan['target_pos'],
            object_position=self._get_true_object_position(),
        )
        self._log_fusion_warning(before_release_observation)
        self.gripper.open_gripper()
        self.runtime.step(self.config.place.settle_steps)
        after_release_observation = self._observe_stage(
            'after_release',
            planned_position=place_plan['target_pos'],
            object_position=self._get_true_object_position(),
        )
        self._log_fusion_warning(after_release_observation)
        self.arm.move_and_verify(retreat_pos, target_orn=place_plan['target_orn'], label='move_post_place')
        self.arm.move_home()
        self._place_success = True
        self.logger.info('PLACE_FINISH retreat=%s', self._fmt_vec(retreat_pos))
        return {
            'place_plan': place_plan,
            'before_release': before_release_observation,
            'after_release': after_release_observation,
        }

    def run(self, gui: bool = True):
        self.runtime.connect(gui=gui)
        try:
            self._reset_run_metrics()
            self.latest_vision_position = None
            self._last_fusion_control_step = -10**9
            self._fusion_stage_actions = {}
            self._activate_multimodal_systems()
            self.logger.info('Start grasp workflow')
            self.runtime.step(self.config.motion.settle_steps)
            self.arm.move_home()
            after_home_observation = self._observe_stage('after_home', object_position=self._get_true_object_position())
            self._log_fusion_warning(after_home_observation)

            initial_obj_pos = self.vision.locate_object(self.runtime.obj_id)
            refined_obj_pos = self._maybe_refine_object_position(initial_obj_pos)
            self.latest_vision_position = np.array(refined_obj_pos, dtype=float)
            after_vision_observation = self._observe_stage(
                'after_vision_lock',
                vision_position=refined_obj_pos,
                object_position=self._get_true_object_position(),
            )
            self._log_fusion_warning(after_vision_observation)

            candidates = self.arm.iter_grasp_pose_candidates(
                refined_obj_pos,
                z_offset=self.config.motion.grasp_offset,
            )
            if not candidates:
                self._set_failure_reason('no_grasp_pose_candidates')
                self.logger.error('GRASP_FAILED no grasp pose candidates available')
                return self._build_run_result(status='grasp_failed')

            success = False
            chosen_plan = None
            attempt_payload = None

            max_attempts = max(1, int(self.config.motion.max_grasp_candidate_trials))
            for attempt_index, plan in enumerate(candidates[:max_attempts], start=1):
                self._log_target_compare(f'grasp_attempt_{attempt_index}', refined_obj_pos, plan['target_pos'])
                success, attempt_payload = self._execute_grasp_attempt(refined_obj_pos, plan, attempt_index)
                if success:
                    chosen_plan = plan
                    break

            if not success or chosen_plan is None or attempt_payload is None:
                self._set_failure_reason(self._failure_reason or 'no_candidate_achieved_stable_lifted_grasp')
                self.logger.error('GRASP_FAILED no candidate achieved a stable lifted grasp')
                return self._build_run_result(status='grasp_failed')

            self.logger.info('GRASP_SUCCESS yaw=%.3f target=%s', chosen_plan['yaw'], self._fmt_vec(chosen_plan['target_pos']))
            self.runtime.step(self.config.motion.hold_steps)
            self.gripper.reinforce_grip_for_transport()

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
                grip_adjust=self._build_fusion_control_hook('transport_monitor_lift', planned_position=lift_pos),
            )
            self._log_end_effector_state('after_lift', lift_pos)
            after_lift_observation = self._observe_stage(
                'after_lift',
                planned_position=lift_pos,
                object_position=self._get_true_object_position(),
            )
            self._log_fusion_warning(after_lift_observation)

            if self.config.place.enabled:
                self.execute_place()

            self.logger.info('Workflow finished')
            return self._build_run_result(
                status='success',
                chosen_plan=chosen_plan,
                attempt_payload=attempt_payload,
            )
        finally:
            self._deactivate_multimodal_systems()
            self.runtime.disconnect()
