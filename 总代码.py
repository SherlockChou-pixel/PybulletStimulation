import pybullet as p
import pybullet_data
import numpy as np
import time

# ========================
# 配置参数（来自先前代码的精华）
# ========================
pandaEndEffectorIndex = 11
pandaNumDofs = 7

# 真实关节限值（来自 Franka Panda 官方）
ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
ul = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
jr = [ul[i] - ll[i] for i in range(pandaNumDofs)]
rp = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # 更合理的 rest pose

class MultiSensor:
    def __init__(self):
        self.land_id = p.loadURDF("plane.urdf", [0, 0, 0])
        # 加载 Panda 并设置固定基座
        self.panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.obj_id = p.loadURDF("cube_small.urdf", [0.7, 0.2, 0.2])
        # Panda 机械臂仅控制前 7 个关节，夹爪单独控制
        self.arm_joints = list(range(pandaNumDofs))  # [0..6]
        self.gripper_joints = [9, 10]

        # 手眼标定偏置补偿（单位: m）
        # 正值方向: +x 向前, +y 向左, +z 向上
        # 如果抓取总是“偏到同一侧”，调这里即可
        self.cam_to_world_bias = np.array([0.0, 0.0, -0.03])

        # 初始化夹爪张开
        self.current_grip_force = 0.0
        self.current_grip_target = 0.04
        self.open_gripper()


    def get_camera_image_params(self, width=640, height=480):
        cam_pos = [0.60, 0.00, 0.90]
        cam_target_pos = [0.60, 0.00, 0.00]
        cam_up = [0, 1, 0]
        fov, aspect, near, far = 60, width / height, 0.1, 2.0
        view_matrix = p.computeViewMatrix(cam_pos, cam_target_pos, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        w, h, rgba, depth, seg = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return {
            "rgb": rgba, "depth": np.asarray(depth), "seg": np.asarray(seg),
            "view_matrix": view_matrix, "proj_matrix": proj_matrix,
            "width": width, "height": height, "near": near, "far": far
        }

    def process_depth_and_get_position(self, data, target_object_id):
        depth = data["depth"].reshape(data["height"], data["width"])
        seg_raw = data["seg"].reshape(data["height"], data["width"])

        # segmentation mask 的低 24bit 是 objectUniqueId
        seg_obj = seg_raw & ((1 << 24) - 1)
        ys, xs = np.where(seg_obj == target_object_id)
        if len(xs) == 0:
            print(f"未检测到物体 ID {target_object_id}")
            return None

        proj = np.array(data["proj_matrix"]).reshape(4, 4).T
        view = np.array(data["view_matrix"]).reshape(4, 4).T
        inv_vp = np.linalg.inv(proj @ view)

        # 将像素+深度反投影到世界坐标（像素中心 +0.5，减小系统偏差）
        def pixel_to_world(px, py, z_buffer):
            x_ndc = (2.0 * (px + 0.5) / data["width"]) - 1.0
            y_ndc = 1.0 - (2.0 * (py + 0.5) / data["height"])
            z_ndc = 2.0 * z_buffer - 1.0
            ndc = np.array([x_ndc, y_ndc, z_ndc, 1.0])
            world = inv_vp @ ndc
            world /= world[3]
            return world[:3]

        # 采样整块 mask 的 3D 点，比“单个中心像素”更稳
        sample_step = max(1, len(xs) // 1200)
        points = []
        for i in range(0, len(xs), sample_step):
            x, y = xs[i], ys[i]
            z_buffer = depth[y, x]
            points.append(pixel_to_world(x, y, z_buffer))
        points = np.asarray(points)

        # 用物体上表面点估计抓取点（避免取到侧面导致偏差）
        z_threshold = np.percentile(points[:, 2], 85)
        top_points = points[points[:, 2] >= z_threshold]
        grasp_point = np.mean(top_points, axis=0)

        return grasp_point

    def set_gripper(self, target_pos, force):
        target_pos = float(np.clip(target_pos, 0.0, 0.04))
        force = float(max(0.0, force))
        self.current_grip_target = target_pos
        self.current_grip_force = force
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.panda_id, j, p.POSITION_CONTROL, target_pos, force=force)

    def open_gripper(self, target_pos=0.04, force=80):
        self.set_gripper(target_pos, force)

    def close_gripper(self, target_pos=0.01, force=120):
        self.set_gripper(target_pos, force)

    def get_gripper_contact_state(self, object_id=None):
        if object_id is None:
            object_id = self.obj_id

        contact_count = 0
        total_normal_force = 0.0
        active_fingers = 0

        for joint_id in self.gripper_joints:
            contacts = p.getContactPoints(
                bodyA=self.panda_id,
                bodyB=object_id,
                linkIndexA=joint_id
            )
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
            object_id = self.obj_id

        if base_force is None:
            base_force = self.current_grip_force if self.current_grip_force > 0 else 40
        if max_force is None:
            max_force = max(base_force, self.current_grip_force)
        if target_pos is None:
            target_pos = self.current_grip_target

        contact_state = self.get_gripper_contact_state(object_id)
        total_normal_force = contact_state["total_normal_force"]
        active_fingers = contact_state["active_fingers"]
        new_force = self.current_grip_force if self.current_grip_force > 0 else base_force

        if active_fingers < 2 or total_normal_force < 0.5:
            new_force = min(max_force, new_force + 2.0)
        elif total_normal_force > 8.0:
            new_force = max(base_force, new_force - 1.0)

        self.set_gripper(target_pos, new_force)
        return new_force, contact_state

    def estimate_gripper_force(self, object_id=None):
        """
        根据物体质量和尺寸，估算夹爪闭合时需要的目标力矩、最大驱动力和最小闭合位置。
        """
        if object_id is None:
            object_id = self.obj_id

        mass = 0.1
        try:
            mass = p.getDynamicsInfo(object_id, -1)[0]
        except Exception:
            pass

        if mass <= 0:
            mass = 0.1

        aabb_min, aabb_max = p.getAABB(object_id)
        object_size = np.array(aabb_max) - np.array(aabb_min)

        horizontal_span = max(0.01, float(min(object_size[0], object_size[1])))
        finger_target_pos = np.clip(horizontal_span * 0.5 - 0.003, 0.005, 0.04)

        gravity_force = mass * 9.8
        size_factor = np.clip(horizontal_span / 0.05, 0.6, 1.6)

        # 这里使用你实测得到的力尺度：夹紧通常在 0.05 左右。
        # 因此动态调节只在这个量级附近做小范围变化，而不是之前错误的大量级。
        target_torque = np.clip(0.035 + gravity_force * 0.008 + (size_factor - 1.0) * 0.010, 0.030, 0.080)
        max_motor_force = np.clip(target_torque * 1.20, 0.040, 0.100)
        hold_force = np.clip(target_torque * 1.05, 0.035, 0.090)

        return {
            "mass": mass,
            "object_size": object_size,
            "target_torque": float(target_torque),
            "max_motor_force": float(max_motor_force),
            "hold_force": float(hold_force),
            "finger_target_pos": float(finger_target_pos),
        }

    def rotate_gripper(self, angle_rad):
        p.setJointMotorControl2(
            bodyIndex=self.panda_id,
            jointIndex=6,  # ← 关键：手腕旋转关节
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_rad,
            force=100,
            maxVelocity=1.0
        )
    def close_gripper_with_force_control(self, target_force=None, max_steps=500, object_id=None):
        """
        自适应力控夹爪闭合。
        优先根据物体质量和尺寸动态估算夹持参数，也支持手动覆盖 target_force。
        """
        print("【力控夹爪】开始自适应闭合...")

        if object_id is None:
            object_id = self.obj_id

        grasp_cfg = self.estimate_gripper_force(object_id)
        if target_force is None:
            target_force = grasp_cfg["target_torque"]
        target_force = max(0.0, float(target_force))

        max_motor_force = max(20.0, grasp_cfg["max_motor_force"] * 1000.0)
        hold_force = max(12.0, grasp_cfg["hold_force"] * 1000.0)
        hold_force = float(np.clip(max(hold_force, target_force * 1000.0), 12.0, max_motor_force))
        min_finger_pos = grasp_cfg["finger_target_pos"]

        print(
            f"【力控夹爪】质量={grasp_cfg['mass']:.3f}kg, "
            f"尺寸={np.round(grasp_cfg['object_size'], 3)}, "
            f"目标力矩={target_force:.4f}, 最大驱动力={max_motor_force:.4f}"
        )

        # 先稍微张开一点，确保从松开状态开始
        self.open_gripper()
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240.)

        open_pos = 0.04
        close_steps = max(20, min(max_steps, 80))
        final_pos = min_finger_pos

        for step in range(close_steps):
            ratio = (step + 1) / close_steps
            target_pos_now = open_pos + (min_finger_pos - open_pos) * ratio
            motor_force_now = hold_force + (max_motor_force - hold_force) * ratio
            self.set_gripper(target_pos_now, motor_force_now)

            for _ in range(4):
                p.stepSimulation()
                time.sleep(1 / 240.)

            contact_state = self.get_gripper_contact_state(object_id)
            if contact_state["active_fingers"] >= 2 and contact_state["total_normal_force"] > 0.5:
                joint_states = [p.getJointState(self.panda_id, joint_id)[0] for joint_id in self.gripper_joints]
                final_pos = max(min_finger_pos, float(np.mean(joint_states)))
                break

        self.set_gripper(final_pos, hold_force)

        for _ in range(90):
            _, contact_state = self.adjust_grip_force(
                object_id=object_id,
                base_force=hold_force,
                max_force=max_motor_force,
                target_pos=final_pos,
            )
            p.stepSimulation()
            time.sleep(1 / 240.)

        print(
            f"【力控夹爪】保持夹持，final_pos={final_pos:.4f}, "
            f"hold_force={self.current_grip_force:.2f}, "
            f"contact_force={contact_state['total_normal_force']:.4f}"
        )
        return {
            "final_pos": final_pos,
            "hold_force": hold_force,
            "max_motor_force": max_motor_force,
        }
    def move_arm_to_position(self, target_pos, target_orn=None, steps=240, grip_adjust=None):
        # 默认让夹爪朝下（z 轴向下），适合竖直抓取
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        joint_poses = p.calculateInverseKinematics(
            self.panda_id,
            pandaEndEffectorIndex,
            target_pos,
            target_orn,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        # 仅给前 7 个关节下发目标，避免把手指关节也当成机械臂关节控制
        for j in range(pandaNumDofs):
            p.setJointMotorControl2(
                bodyIndex=self.panda_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=500,
                maxVelocity=1.0,
                positionGain=0.1,
                velocityGain=1.0
            )

        # 给到位时间
        self.wait(steps, grip_adjust=grip_adjust)

    def get_obj_pos_by_depth(self):
        data = self.get_camera_image_params()
        calc_pos = self.process_depth_and_get_position(data, self.obj_id)
        true_pos, _ = p.getBasePositionAndOrientation(self.obj_id)

        if calc_pos is not None:
            # 应用手眼标定偏置补偿
            corrected_pos = np.array(calc_pos) + self.cam_to_world_bias

            raw_err = np.linalg.norm(np.array(true_pos) - np.array(calc_pos))
            corrected_err = np.linalg.norm(np.array(true_pos) - corrected_pos)
            print(f"[视觉定位] 原始: {calc_pos}, 补偿后: {corrected_pos}, 真实: {true_pos}")
            print(f"[视觉定位] 原始误差: {raw_err:.3f}m, 补偿后误差: {corrected_err:.3f}m, bias: {self.cam_to_world_bias}")

            # 返回补偿后的视觉坐标
            return corrected_pos

        # 回退：如果视觉没识别出来，至少还能用真值继续流程（仿真调试用）
        print("[视觉定位] 使用真值回退")
        return np.array(true_pos)

    def wait(self, steps=60, grip_adjust=None):
        for _ in range(steps):
            if grip_adjust is not None:
                grip_adjust()
            p.stepSimulation()
            time.sleep(1/240.)

    def main(self):
        print("开始仿真...")
        self.wait(60)  # 初始稳定

        # === 状态 1: 定位物体 ===
        print("【状态1】通过深度相机定位物体...")
        obj_pos = self.get_obj_pos_by_depth()
        if obj_pos is None:
            print("定位失败！")
            return

        # 抓取姿态：夹爪朝下；偏航可按目标方位调整
        grasp_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.18]  # 上方悬停
        self.open_gripper()
        self.move_arm_to_position(pre_grasp_pos, target_orn=grasp_orn, steps=240)

        # === 状态 2: 下降至抓取位 ===
        print("【状态2】下降至抓取位置...")
        grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02]  # 给一点高度裕量，避免碰撞
        self.move_arm_to_position(grasp_pos, target_orn=grasp_orn, steps=240)

        # === 状态 3: 闭合夹爪 ===
        print("【状态3】闭合夹爪...")
        grip_state = self.close_gripper_with_force_control()
        self.wait(
            180,
            grip_adjust=lambda: self.adjust_grip_force(
                object_id=self.obj_id,
                base_force=grip_state["hold_force"],
                max_force=grip_state["max_motor_force"],
                target_pos=grip_state["final_pos"],
            )
        )

        # === 状态 4: 提升物体 ===
        print("【状态4】提升物体...")
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.25]
        self.move_arm_to_position(
            lift_pos,
            target_orn=grasp_orn,
            steps=240,
            grip_adjust=lambda: self.adjust_grip_force(
                object_id=self.obj_id,
                base_force=grip_state["hold_force"],
                max_force=grip_state["max_motor_force"],
                target_pos=grip_state["final_pos"],
            )
        )

        print("✅ 抓取任务完成！")

if __name__ == '__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=60,
        cameraPitch=-25,
        cameraTargetPosition=[0.6, 0.0, 0.3],
    )
    ms = MultiSensor()
    try:
        ms.main()
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
