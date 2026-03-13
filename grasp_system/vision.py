import numpy as np
import pybullet as p


class VisionSystem:
    def __init__(self, runtime, config, logger):
        self.runtime = runtime
        self.config = config
        self.logger = logger

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
            "rgb": rgba,
            "depth": np.asarray(depth),
            "seg": np.asarray(seg),
            "view_matrix": view_matrix,
            "proj_matrix": proj_matrix,
            "width": camera.width,
            "height": camera.height,
        }

    def process_depth_and_get_position(self, data, target_object_id):
        depth = data["depth"].reshape(data["height"], data["width"])
        seg_raw = data["seg"].reshape(data["height"], data["width"])
        seg_obj = seg_raw & ((1 << 24) - 1)
        ys, xs = np.where(seg_obj == target_object_id)
        if len(xs) == 0:
            self.logger.warning("深度相机未检测到目标物体，object_id=%s", target_object_id)
            return None

        proj = np.array(data["proj_matrix"]).reshape(4, 4).T
        view = np.array(data["view_matrix"]).reshape(4, 4).T
        inv_vp = np.linalg.inv(proj @ view)

        def pixel_to_world(px, py, z_buffer):
            x_ndc = (2.0 * (px + 0.5) / data["width"]) - 1.0
            y_ndc = 1.0 - (2.0 * (py + 0.5) / data["height"])
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

        points = np.asarray(points)
        z_threshold = np.percentile(points[:, 2], 85)
        top_points = points[points[:, 2] >= z_threshold]
        return np.mean(top_points, axis=0)

    def locate_object(self, object_id):
        data = self.capture()
        calc_pos = self.process_depth_and_get_position(data, object_id)
        true_pos, _ = p.getBasePositionAndOrientation(object_id)

        if calc_pos is None:
            self.logger.warning("视觉定位失败，回退到真实位置 %s", np.round(true_pos, 3))
            return np.array(true_pos)

        corrected_pos = np.array(calc_pos) + self.config.scene.cam_to_world_bias
        raw_err = np.linalg.norm(np.array(true_pos) - np.array(calc_pos))
        corrected_err = np.linalg.norm(np.array(true_pos) - corrected_pos)
        self.logger.info(
            "视觉定位完成 raw=%s corrected=%s true=%s raw_err=%.4f corrected_err=%.4f",
            np.round(calc_pos, 4),
            np.round(corrected_pos, 4),
            np.round(true_pos, 4),
            raw_err,
            corrected_err,
        )
        return corrected_pos
