import numpy as np


class MultiModalObserver:
    def __init__(self, runtime, imu, pressure, config, logger):
        self.runtime = runtime
        self.imu = imu
        self.pressure = pressure
        self.config = config
        self.logger = logger
        self._last_vision_confidence = 0.5
        self._last_fusion_snapshot = None

    @staticmethod
    def _fmt_vec(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    @staticmethod
    def _clamp01(value):
        return float(np.clip(value, 0.0, 1.0))

    def _score_from_error(self, error_value, good_error, max_error):
        if error_value <= good_error:
            return 1.0
        if error_value >= max_error:
            return 0.0
        span = max(max_error - good_error, 1e-6)
        return self._clamp01(1.0 - (error_value - good_error) / span)

    @staticmethod
    def _is_transport_stage(stage):
        return (
            'transport_monitor' in stage
            or 'after_validation_lift' in stage
            or stage in ('after_lift', 'before_release')
        )

    def _compute_vision_metrics(self, stage, vision_position=None, object_position=None):
        transport_stage = self._is_transport_stage(stage) or stage == 'after_release'
        if transport_stage:
            return {
                'confidence': float(self._last_vision_confidence),
                'error': None,
                'source': 'frozen_for_transport',
            }
        if vision_position is None:
            return {
                'confidence': float(self._last_vision_confidence),
                'error': None,
                'source': 'carry_over',
            }

        if object_position is None:
            confidence = float(self._last_vision_confidence)
            return {
                'confidence': confidence,
                'error': None,
                'source': 'carry_over',
            }

        error = float(np.linalg.norm(np.array(vision_position, dtype=float) - np.array(object_position, dtype=float)))
        confidence = self._score_from_error(
            error,
            self.config.fusion.vision_good_error,
            self.config.fusion.vision_max_error,
        )
        self._last_vision_confidence = confidence
        return {
            'confidence': confidence,
            'error': error,
            'source': 'sim_eval',
        }

    def _compute_pressure_metrics(self, pressure):
        if not pressure:
            return {
                'confidence': 0.0,
                'normal_force': 0.0,
                'lateral_force': 0.0,
                'force_balance': 1.0,
                'lateral_ratio': 0.0,
                'active_fingers': 0,
                'stable_contact': False,
                'stable_ratio': 0.0,
            }

        left = pressure.get('per_finger', {}).get('left_finger', {})
        right = pressure.get('per_finger', {}).get('right_finger', {})
        left_force = float(left.get('total_normal_force', 0.0))
        right_force = float(right.get('total_normal_force', 0.0))
        total_normal_force = float(pressure.get('total_normal_force', 0.0))
        total_lateral_force = float(pressure.get('total_lateral_force', 0.0))
        active_fingers = int(pressure.get('active_fingers', 0))
        stable_ratio = float(pressure.get('stable_ratio', 0.0))
        stable_contact = bool(pressure.get('stable_contact', False))

        if active_fingers == 0 or total_normal_force <= 1e-6:
            return {
                'confidence': 0.0,
                'normal_force': total_normal_force,
                'lateral_force': total_lateral_force,
                'force_balance': 0.0,
                'lateral_ratio': 0.0,
                'active_fingers': active_fingers,
                'stable_contact': False,
                'stable_ratio': stable_ratio,
            }

        force_score = self._clamp01(total_normal_force / max(self.config.fusion.desired_total_normal_force, 1e-6))
        contact_score = self._clamp01(active_fingers / max(self.config.pressure.min_active_fingers, 1))
        force_balance = abs(left_force - right_force) / max(total_normal_force, 1e-6)
        balance_score = 1.0 - self._clamp01(force_balance / max(self.config.fusion.force_balance_tolerance, 1e-6))
        lateral_ratio = total_lateral_force / max(total_normal_force, 1e-6)
        friction_score = 1.0 - self._clamp01(lateral_ratio / max(self.config.fusion.lateral_ratio_warn, 1e-6))
        stable_score = stable_ratio if stable_contact else 0.5 * stable_ratio

        confidence = (
            0.30 * force_score
            + 0.20 * contact_score
            + 0.20 * stable_score
            + 0.15 * balance_score
            + 0.15 * friction_score
        )
        if stable_contact:
            confidence = min(1.0, confidence + 0.05)

        return {
            'confidence': self._clamp01(confidence),
            'normal_force': total_normal_force,
            'lateral_force': total_lateral_force,
            'force_balance': force_balance,
            'lateral_ratio': lateral_ratio,
            'active_fingers': active_fingers,
            'stable_contact': stable_contact,
            'stable_ratio': stable_ratio,
        }

    def _compute_imu_metrics(self, imu):
        ee_imu = (imu or {}).get('end_effector') or {}
        object_imu = (imu or {}).get('object') or {}

        obj_speed = float(object_imu.get('latest_linear_speed', 0.0))
        obj_peak_acc = float(object_imu.get('peak_linear_acceleration', 0.0))
        ee_peak_acc = float(ee_imu.get('peak_linear_acceleration', 0.0))

        object_speed_score = 1.0 - self._clamp01(obj_speed / max(self.config.fusion.object_speed_stable, 1e-6))
        object_acc_score = 1.0 - self._clamp01(obj_peak_acc / max(self.config.fusion.object_peak_acc_stable, 1e-6))
        ee_smooth_score = 1.0 - self._clamp01(ee_peak_acc / max(self.config.fusion.end_effector_peak_acc_smooth, 1e-6))

        confidence = 0.50 * object_speed_score + 0.35 * object_acc_score + 0.15 * ee_smooth_score
        return {
            'confidence': self._clamp01(confidence),
            'object_speed': obj_speed,
            'object_peak_acc': obj_peak_acc,
            'ee_peak_acc': ee_peak_acc,
        }

    def _compute_slip_risk(self, stage, pressure_metrics, imu_metrics):
        prev_snapshot = self._last_fusion_snapshot or {}
        prev_normal_force = float(prev_snapshot.get('pressure_normal', pressure_metrics['normal_force']))
        contact_expected = any(
            key in stage
            for key in (
                'after_close',
                'after_validation_lift',
                'after_lift',
                'before_release',
                'transport_monitor',
            )
        )
        if not contact_expected and pressure_metrics['active_fingers'] == 0:
            return {
                'score': 0.0,
                'pressure_drop_ratio': 0.0,
            }
        if prev_normal_force <= 1e-6:
            pressure_drop_ratio = 0.0
        else:
            pressure_drop_ratio = max(prev_normal_force - pressure_metrics['normal_force'], 0.0) / prev_normal_force

        pressure_penalty = 1.0 - pressure_metrics['confidence']
        balance_penalty = self._clamp01(
            pressure_metrics['force_balance'] / max(self.config.fusion.force_balance_tolerance, 1e-6)
        )
        lateral_penalty = self._clamp01(
            pressure_metrics['lateral_ratio'] / max(self.config.fusion.lateral_ratio_warn, 1e-6)
        )
        motion_penalty = self._clamp01(
            imu_metrics['object_speed'] / max(self.config.fusion.slip_object_speed_warn, 1e-6)
        )
        if self._is_transport_stage(stage) and pressure_metrics['stable_contact']:
            motion_penalty *= self.config.fusion.transport_motion_penalty_scale
            if (
                pressure_metrics['normal_force'] >= 0.9 * self.config.fusion.desired_total_normal_force
                and pressure_metrics['lateral_ratio'] <= self.config.fusion.control_transport_lateral_ratio
                and pressure_metrics['force_balance'] <= self.config.fusion.force_balance_tolerance
            ):
                motion_penalty *= 0.6
        pressure_drop_penalty = self._clamp01(
            pressure_drop_ratio / max(self.config.fusion.pressure_drop_warn_ratio, 1e-6)
        )

        slip_risk = (
            0.35 * pressure_penalty
            + 0.20 * balance_penalty
            + 0.20 * lateral_penalty
            + 0.15 * motion_penalty
            + 0.10 * pressure_drop_penalty
        )

        if 'after_release' in stage:
            slip_risk = 0.0
        elif 'after_close' in stage and pressure_metrics['stable_contact']:
            slip_risk *= 0.8

        return {
            'score': self._clamp01(slip_risk),
            'pressure_drop_ratio': pressure_drop_ratio,
        }

    def _compute_grasp_confidence(self, stage, vision_metrics, pressure_metrics, imu_metrics):
        vision_confidence = vision_metrics['confidence']
        pressure_confidence = pressure_metrics['confidence']
        imu_confidence = imu_metrics['confidence']

        if stage == 'after_home':
            return 0.0
        if 'after_vision_lock' in stage:
            return min(0.75, self._clamp01(0.65 * vision_confidence + 0.10 * imu_confidence))
        if 'after_descend' in stage:
            return min(0.82, self._clamp01(0.45 * vision_confidence + 0.15 * pressure_confidence + 0.20 * imu_confidence))
        if 'after_close' in stage:
            return self._clamp01(0.20 * vision_confidence + 0.55 * pressure_confidence + 0.25 * imu_confidence)
        if 'after_validation_lift' in stage or 'after_lift' in stage or 'before_release' in stage:
            confidence = self._clamp01(0.72 * pressure_confidence + 0.28 * imu_confidence)
            if (
                pressure_metrics['stable_contact']
                and pressure_metrics['lateral_ratio'] <= self.config.fusion.control_transport_lateral_ratio
                and pressure_metrics['force_balance'] <= self.config.fusion.force_balance_tolerance
            ):
                confidence = max(
                    confidence,
                    self._clamp01(0.66 + 0.18 * pressure_confidence + self.config.fusion.transport_stable_grasp_boost),
                )
            return confidence
        if 'after_release' in stage:
            return 0.0
        if 'transport_monitor' in stage:
            confidence = self._clamp01(0.74 * pressure_confidence + 0.26 * imu_confidence)
            if (
                pressure_metrics['stable_contact']
                and pressure_metrics['normal_force'] >= 0.9 * self.config.fusion.desired_total_normal_force
                and pressure_metrics['lateral_ratio'] <= self.config.fusion.control_transport_lateral_ratio
            ):
                confidence = max(
                    confidence,
                    self._clamp01(0.64 + 0.20 * pressure_confidence + self.config.fusion.transport_stable_grasp_boost),
                )
            return confidence
        return self._clamp01(0.30 * vision_confidence + 0.40 * pressure_confidence + 0.30 * imu_confidence)

    def _infer_fusion_state(self, stage, vision_metrics, pressure_metrics, imu_metrics, slip_metrics, grasp_confidence):
        slip_risk = slip_metrics['score']
        pressure_stable = pressure_metrics['stable_contact']

        if stage == 'after_home':
            return 'HOMED'
        if stage == 'after_vision_lock':
            return 'VISION_LOCKED' if vision_metrics['confidence'] >= 0.7 else 'VISION_UNCERTAIN'
        if 'after_descend' in stage:
            return 'ALIGNING_FOR_GRASP' if pressure_metrics['active_fingers'] == 0 else 'CONTACTING'
        if 'after_close' in stage:
            if pressure_stable and grasp_confidence >= self.config.fusion.stable_grasp_confidence:
                return 'CONTACT_ESTABLISHED'
            return 'CONTACT_UNCERTAIN'
        if 'after_validation_lift' in stage or 'after_lift' in stage:
            if slip_risk >= self.config.fusion.high_slip_risk:
                return 'SLIP_RISK'
            if pressure_stable and grasp_confidence >= self.config.fusion.stable_grasp_confidence:
                return 'GRASP_STABLE'
            return 'GRASP_UNCERTAIN'
        if 'transport_monitor' in stage:
            if pressure_metrics['active_fingers'] == 0:
                return 'GRASP_LOST'
            if slip_risk >= self.config.fusion.high_slip_risk:
                return 'SLIP_RISK'
            if pressure_stable and grasp_confidence >= self.config.fusion.stable_grasp_confidence:
                return 'TRANSPORT_STABLE'
            return 'TRANSPORT_UNCERTAIN'
        if stage == 'before_release':
            if pressure_stable and grasp_confidence >= self.config.fusion.stable_grasp_confidence:
                return 'TRANSPORT_STABLE'
            return 'TRANSPORT_UNCERTAIN'
        if stage == 'after_release':
            return 'RELEASED' if pressure_metrics['normal_force'] <= 1e-3 else 'RELEASE_UNCERTAIN'
        if imu_metrics['object_peak_acc'] > self.config.fusion.object_peak_acc_stable:
            return 'DYNAMIC_TRANSIENT'
        return 'MONITORING'

    def _build_alerts(self, stage, vision_metrics, pressure_metrics, imu_metrics, slip_metrics, grasp_confidence):
        alerts = []
        contact_expected = any(
            key in stage
            for key in (
                'after_close',
                'after_validation_lift',
                'after_lift',
                'before_release',
                'transport_monitor',
            )
        )
        vision_relevant = any(key in stage for key in ('after_vision_lock', 'after_descend', 'after_close'))
        if vision_relevant and vision_metrics['error'] is not None and vision_metrics['confidence'] < 0.7:
            alerts.append(f'vision_error={vision_metrics["error"]:.4f}')
        if contact_expected and pressure_metrics['active_fingers'] < self.config.pressure.min_active_fingers:
            alerts.append('insufficient_finger_contact')
        if contact_expected and pressure_metrics['stable_contact'] and pressure_metrics['force_balance'] > self.config.fusion.force_balance_tolerance:
            alerts.append(f'force_imbalance={pressure_metrics["force_balance"]:.3f}')
        if contact_expected and pressure_metrics['stable_contact'] and pressure_metrics['lateral_ratio'] > self.config.fusion.lateral_ratio_warn:
            alerts.append(f'lateral_ratio={pressure_metrics["lateral_ratio"]:.3f}')
        if (
            contact_expected
            and
            pressure_metrics['stable_contact']
            and imu_metrics['object_speed'] > self.config.fusion.slip_object_speed_warn
            and 'after_release' not in stage
        ):
            alerts.append(f'object_motion={imu_metrics["object_speed"]:.4f}')
        if (
            contact_expected
            and slip_metrics['pressure_drop_ratio'] > self.config.fusion.pressure_drop_warn_ratio
            and 'after_release' not in stage
        ):
            alerts.append(f'pressure_drop={slip_metrics["pressure_drop_ratio"]:.3f}')
        if contact_expected and grasp_confidence < self.config.fusion.stable_grasp_confidence:
            alerts.append(f'low_grasp_confidence={grasp_confidence:.3f}')
        return alerts

    def _compute_fusion(self, stage, observation):
        vision_metrics = self._compute_vision_metrics(
            stage,
            observation.get('vision_position'),
            observation.get('object_position'),
        )
        pressure_metrics = self._compute_pressure_metrics(observation.get('pressure'))
        imu_metrics = self._compute_imu_metrics(observation.get('imu'))
        slip_metrics = self._compute_slip_risk(stage, pressure_metrics, imu_metrics)
        grasp_confidence = self._compute_grasp_confidence(
            stage,
            vision_metrics,
            pressure_metrics,
            imu_metrics,
        )
        fusion_state = self._infer_fusion_state(
            stage,
            vision_metrics,
            pressure_metrics,
            imu_metrics,
            slip_metrics,
            grasp_confidence,
        )
        alerts = self._build_alerts(
            stage,
            vision_metrics,
            pressure_metrics,
            imu_metrics,
            slip_metrics,
            grasp_confidence,
        )

        fusion = {
            'state': fusion_state,
            'grasp_confidence': grasp_confidence,
            'slip_risk': slip_metrics['score'],
            'vision_confidence': vision_metrics['confidence'],
            'vision_error': vision_metrics['error'],
            'pressure_confidence': pressure_metrics['confidence'],
            'imu_stability_score': imu_metrics['confidence'],
            'pressure_normal': pressure_metrics['normal_force'],
            'pressure_lateral_ratio': pressure_metrics['lateral_ratio'],
            'pressure_balance': pressure_metrics['force_balance'],
            'pressure_drop_ratio': slip_metrics['pressure_drop_ratio'],
            'object_speed': imu_metrics['object_speed'],
            'object_peak_acc': imu_metrics['object_peak_acc'],
            'alerts': alerts,
        }
        self._last_fusion_snapshot = fusion
        return fusion

    def capture(self, stage, vision_position=None, planned_position=None, object_position=None):
        observation = {
            'stage': stage,
            'sim_time': float(self.runtime.sim_time),
            'imu': {
                'end_effector': self.imu.build_summary('end_effector'),
                'object': self.imu.build_summary('object'),
            },
            'pressure': self.pressure.build_summary(),
        }
        if vision_position is not None:
            observation['vision_position'] = self._fmt_vec(vision_position)
        if planned_position is not None:
            observation['planned_position'] = self._fmt_vec(planned_position)
        if object_position is not None:
            observation['object_position'] = self._fmt_vec(object_position)
        if self.config.fusion.enabled:
            observation['fusion'] = self._compute_fusion(stage, observation)
        return observation

    def log_stage(self, stage, vision_position=None, planned_position=None, object_position=None):
        observation = self.capture(
            stage,
            vision_position=vision_position,
            planned_position=planned_position,
            object_position=object_position,
        )
        pressure = observation.get('pressure') or {}
        ee_imu = (observation.get('imu') or {}).get('end_effector') or {}
        object_imu = (observation.get('imu') or {}).get('object') or {}

        parts = [
            f'stage={stage}',
            f'sim_time={observation["sim_time"]:.4f}',
        ]
        if 'vision_position' in observation:
            parts.append(f'vision={observation["vision_position"]}')
        if 'planned_position' in observation:
            parts.append(f'planned={observation["planned_position"]}')
        if 'object_position' in observation:
            parts.append(f'object={observation["object_position"]}')
        if ee_imu:
            parts.append(f'ee_speed={ee_imu["latest_linear_speed"]:.4f}')
            parts.append(f'ee_peak_acc={ee_imu["peak_linear_acceleration"]:.4f}')
        if object_imu:
            parts.append(f'obj_speed={object_imu["latest_linear_speed"]:.4f}')
            parts.append(f'obj_peak_acc={object_imu["peak_linear_acceleration"]:.4f}')
        if pressure:
            parts.append(f'pressure_normal={pressure["total_normal_force"]:.4f}')
            parts.append(f'pressure_active_fingers={pressure["active_fingers"]}')
            parts.append(f'pressure_stable={pressure["stable_contact"]}')
            left = pressure['per_finger'].get('left_finger', {})
            right = pressure['per_finger'].get('right_finger', {})
            parts.append(f'left_force={left.get("total_normal_force", 0.0):.4f}')
            parts.append(f'right_force={right.get("total_normal_force", 0.0):.4f}')

        self.logger.info('MULTIMODAL %s', ' | '.join(parts))
        fusion = observation.get('fusion') or {}
        if fusion:
            fusion_parts = [
                f'stage={stage}',
                f'state={fusion["state"]}',
                f'grasp_confidence={fusion["grasp_confidence"]:.3f}',
                f'slip_risk={fusion["slip_risk"]:.3f}',
                f'vision_confidence={fusion["vision_confidence"]:.3f}',
                f'pressure_confidence={fusion["pressure_confidence"]:.3f}',
                f'imu_stability={fusion["imu_stability_score"]:.3f}',
                f'pressure_balance={fusion["pressure_balance"]:.3f}',
                f'pressure_lateral_ratio={fusion["pressure_lateral_ratio"]:.3f}',
                f'pressure_drop_ratio={fusion["pressure_drop_ratio"]:.3f}',
                f'object_speed={fusion["object_speed"]:.4f}',
                f'object_peak_acc={fusion["object_peak_acc"]:.4f}',
            ]
            if fusion.get('vision_error') is not None:
                fusion_parts.append(f'vision_error={fusion["vision_error"]:.4f}')
            if fusion.get('alerts'):
                fusion_parts.append(f'alerts={fusion["alerts"]}')
            self.logger.info('FUSION_STATE %s', ' | '.join(fusion_parts))
        return observation
