import numpy as np
import pybullet as p


class VisionSystem:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger
        self.cam_to_world_bias = np.array(self.config.scene.cam_to_world_bias, dtype=float)

    @staticmethod
    def _fmt_vec(vec):
        return np.round(np.array(vec, dtype=float), 4).tolist()

    def capture(self):
        camera = self.config.camera
        aspect = camera.width / camera.height
        view_matrix = p.computeViewMatrix(camera.position, camera.target, camera.up)
        proj_matrix = p.computeProjectionMatrixFOV(camera.fov, aspect, camera.near, camera.far)
        _, _, rgba, depth, seg = p.getCameraImage(
            camera.width,
            camera.height,
            view_matrix,
            proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return {
            'rgb': rgba,
            'depth': np.asarray(depth),
            'seg': np.asarray(seg),
            'view_matrix': view_matrix,
            'proj_matrix': proj_matrix,
            'width': camera.width,
            'height': camera.height,
        }

    def process_depth_and_get_position(self, data, target_object_id):
        depth = data['depth'].reshape(data['height'], data['width'])
        seg_raw = data['seg'].reshape(data['height'], data['width'])
        seg_obj = seg_raw & ((1 << 24) - 1)
        ys, xs = np.where(seg_obj == target_object_id)
        if len(xs) == 0:
            self.logger.warning('VISION_FRAME miss object_id=%s', target_object_id)
            return None

        proj = np.array(data['proj_matrix']).reshape(4, 4).T
        view = np.array(data['view_matrix']).reshape(4, 4).T
        inv_vp = np.linalg.inv(proj @ view)

        def pixel_to_world(px, py, z_buffer):
            x_ndc = (2.0 * (px + 0.5) / data['width']) - 1.0
            y_ndc = 1.0 - (2.0 * (py + 0.5) / data['height'])
            z_ndc = 2.0 * z_buffer - 1.0
            ndc = np.array([x_ndc, y_ndc, z_ndc, 1.0])
            world = inv_vp @ ndc
            world /= world[3]
            return world[:3]

        sample_step = max(1, len(xs) // 1200)
        points = []
        for index in range(0, len(xs), sample_step):
            x, y = xs[index], ys[index]
            points.append(pixel_to_world(x, y, depth[y, x]))

        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return None

        top_percentile = float(self.config.vision.top_percentile)
        low_percentile = float(self.config.vision.center_low_percentile)
        high_percentile = float(self.config.vision.center_high_percentile)
        top_band = float(max(0.0, self.config.vision.top_surface_band))

        z_top = float(np.percentile(points[:, 2], top_percentile))
        z_bottom = float(np.percentile(points[:, 2], 100.0 - top_percentile))
        top_points = points[points[:, 2] >= z_top - top_band]
        if len(top_points) == 0:
            top_points = points[points[:, 2] >= np.percentile(points[:, 2], 95.0)]
        if len(top_points) == 0:
            top_points = points

        x_low, x_high = np.percentile(top_points[:, 0], [low_percentile, high_percentile])
        y_low, y_high = np.percentile(top_points[:, 1], [low_percentile, high_percentile])
        x_center = 0.5 * (x_low + x_high)
        y_center = 0.5 * (y_low + y_high)
        span_x = float(max(0.0, x_high - x_low))
        span_y = float(max(0.0, y_high - y_low))
        inferred_height = max(0.0, min(span_x, span_y))
        if inferred_height > 1e-4:
            z_center = z_top - 0.5 * inferred_height
        else:
            z_center = 0.5 * (z_top + z_bottom)

        return np.array([x_center, y_center, z_center], dtype=float)

    def locate_object(self, object_id):
        true_pos, _ = p.getBasePositionAndOrientation(object_id)
        true_pos = np.array(true_pos, dtype=float)
        raw_positions = []
        corrected_positions = []

        for frame_index in range(max(1, self.config.vision.capture_repeats)):
            data = self.capture()
            calc_pos = self.process_depth_and_get_position(data, object_id)
            if calc_pos is None:
                continue

            raw_pos = np.array(calc_pos, dtype=float)
            corrected_pos = raw_pos + self.cam_to_world_bias

            if self.config.vision.auto_bias_calibration_in_sim:
                target_bias = true_pos - raw_pos
                alpha = float(np.clip(self.config.vision.bias_update_alpha, 0.0, 1.0))
                updated_bias = (1.0 - alpha) * self.cam_to_world_bias + alpha * target_bias
                self.logger.info(
                    'VISION_BIAS_UPDATE previous=%s target=%s updated=%s',
                    self._fmt_vec(self.cam_to_world_bias),
                    self._fmt_vec(target_bias),
                    self._fmt_vec(updated_bias),
                )
                self.cam_to_world_bias = updated_bias
                corrected_pos = raw_pos + self.cam_to_world_bias

            raw_positions.append(raw_pos)
            corrected_positions.append(corrected_pos)

            if self.config.logging.log_vision_frames:
                self.logger.info(
                    'VISION_FRAME frame=%d camera_raw=%s corrected=%s api_true=%s raw_delta=%s corrected_delta=%s',
                    frame_index + 1,
                    self._fmt_vec(raw_pos),
                    self._fmt_vec(corrected_pos),
                    self._fmt_vec(true_pos),
                    self._fmt_vec(raw_pos - true_pos),
                    self._fmt_vec(corrected_pos - true_pos),
                )
            self.runtime.step(1)

        if not corrected_positions:
            self.logger.warning('VISION_SUMMARY fallback_to_true api_true=%s', self._fmt_vec(true_pos))
            return true_pos

        raw_pos = np.median(np.asarray(raw_positions), axis=0)
        corrected_pos = np.median(np.asarray(corrected_positions), axis=0)
        raw_err = np.linalg.norm(true_pos - raw_pos)
        corrected_err = np.linalg.norm(true_pos - corrected_pos)
        self.logger.info(
            'VISION_SUMMARY camera_raw=%s corrected=%s api_true=%s raw_delta=%s corrected_delta=%s raw_err=%.4f corrected_err=%.4f repeats=%d',
            self._fmt_vec(raw_pos),
            self._fmt_vec(corrected_pos),
            self._fmt_vec(true_pos),
            self._fmt_vec(raw_pos - true_pos),
            self._fmt_vec(corrected_pos - true_pos),
            raw_err,
            corrected_err,
            len(corrected_positions),
        )
        return corrected_pos
