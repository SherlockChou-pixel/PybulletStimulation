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
        total_lateral_friction = 0.0
        total_spinning_friction = 0.0
        active_fingers = 0
        friction_coefficients = []
        
        for joint_id in self.gripper_joints:
            contacts = p.getContactPoints(bodyA=self.panda_id, bodyB=object_id, linkIndexA=joint_id)
            if contacts:
                active_fingers += 1
            for contact in contacts:
                # 获取接触点的正压力
                total_normal_force += contact[9]  # normal force is at index 9
                
                # 获取摩擦系数 (index 13)
                if len(contact) > 13:
                    friction_coeff = contact[13]  # lateral friction coefficient is at index 13
                    friction_coefficients.append(friction_coeff)
                
                # 累计横向摩擦力 (indices 10 and 12)
                if len(contact) > 10:
                    total_lateral_friction += contact[10]  # lateral friction force 1 at index 10
                if len(contact) > 12:
                    total_lateral_friction += contact[12]  # lateral friction force 2 at index 12
                
                # 累计旋转摩擦力 (index 14, if available)
                if len(contact) > 14:
                    total_spinning_friction += contact[14]  # spinning friction at index 14

                contact_count += 1

        avg_friction_coeff = np.mean(friction_coefficients) if friction_coefficients else 0.0
        
        return {
            "contact_count": contact_count,
            "active_fingers": active_fingers,
            "total_normal_force": float(total_normal_force),
            "total_lateral_friction": float(total_lateral_friction),
            "total_spinning_friction": float(total_spinning_friction),
            "avg_friction_coefficient": float(avg_friction_coeff),
            "friction_coefficients": friction_coefficients
        }

    def calculate_min_grasp_force(self, object_id=None):
        """
        根据物体质量与摩擦系数动态计算最小所需抓取力（考虑20%安全余量）
        
        Args:
            object_id: 物体ID，若为None则使用当前配置的物体ID
            
        Returns:
            float: 计算得到的最小抓取力（N），若失败则返回0.0
        """
        if object_id is None:
            object_id = self.object_id
            
        try:
            dynamics_info = p.getDynamicsInfo(object_id, -1)
            mass = dynamics_info[0]
            lateral_friction = dynamics_info[1]

            # 类型安全检查
            if not isinstance(mass, (int, float, np.number)):
                mass = float(np.asscalar(mass)) if hasattr(np, 'asscalar') else float(mass[0]) if isinstance(mass, (list, tuple, np.ndarray)) else 0.0
            if not isinstance(lateral_friction, (int, float, np.number)):
                lateral_friction = float(lateral_friction[0]) if isinstance(lateral_friction, (list, tuple, np.ndarray)) else 0.0

            gravity = 9.8
            weight = mass * gravity
            min_normal_force = weight / lateral_friction  # N ≥ G / μ

            # 预留20%余量，防止滑动
            min_grasp_force = min_normal_force * 1.2
            self.logger.info("计算得最小抓取力 %.3f N (基于质量 %.3f kg, 摩擦系数 %.3f)", min_grasp_force, mass, lateral_friction)
            return min_grasp_force
            
        except Exception as e:
            self.logger.warning("无法获取物体动力学参数以计算最小抓取力: %s", str(e))
            return 0.0

    def set_object_friction(self, object_id=None, lateral_friction=0.5, spinning_friction=0.0, rolling_friction=0.0):
        """
        设置物体的摩擦系数，用于验证算法在不同摩擦条件下的表现
        
        Args:
            object_id: 物体ID，若为None则使用当前配置的物体ID
            lateral_friction: 侧向摩擦系数
            spinning_friction: 自旋摩擦系数
            rolling_friction: 滚动摩擦系数
        """
        if object_id is None:
            object_id = self.object_id
            
        try:
            p.changeDynamics(object_id, -1, lateralFriction=lateral_friction, 
                           spinningFriction=spinning_friction, rollingFriction=rolling_friction)
            self.logger.info("已设置物体摩擦系数: 侧向=%.3f, 自旋=%.3f, 滚动=%.3f", 
                           lateral_friction, spinning_friction, rolling_friction)
        except Exception as e:
            self.logger.error("设置物体摩擦系数失败: %s", str(e))

    def close_with_fixed_force(self, object_id=None, fixed_force=None, close_position=None, stabilize_steps=None):
        if object_id is None:
            object_id = self.object_id
        if fixed_force is None:
            # 使用新封装的方法计算最小抓取力
            fixed_force = self.calculate_min_grasp_force(object_id)
            
            # 若计算失败，则回退到配置值
            if fixed_force <= 0:
                fixed_force = self.config.gripper.close_force
                
        if close_position is None:
            close_position = self.config.gripper.default_close_position
        if stabilize_steps is None:
            stabilize_steps = self.config.motion.close_stabilize_steps

        self.logger.info("开始固定力抓取 force=%.2f", fixed_force)

        # 打开夹爪
        self.open_gripper()
        self.runtime.step(60)

        # 闭合夹爪到指定位置和力度
        self.set_gripper(close_position, fixed_force)
        self.logger.info("夹爪闭合至 %.4f 位置，施加力 %.2f", close_position, fixed_force)

        # 等待夹爪闭合
        for step in range(40):  # 给予一定时间让夹爪闭合
            self.runtime.step(1)

        # 获取抓取前的状态
        pre_contact_state = self.get_contact_state(object_id)
        self.logger.info(
            "抓取前状态 - 接触点数量: %d, 总正压力: %.4f, 平均摩擦系数: %.3f",
            pre_contact_state["contact_count"],
            pre_contact_state["total_normal_force"],
            pre_contact_state["avg_friction_coefficient"]
        )

        # 进行稳定步骤
        for step in range(stabilize_steps):
            self.set_gripper(close_position, fixed_force)  # 保持固定位置和力
            self.runtime.step(1)

        # 获取抓取后的状态
        contact_state = self.get_contact_state(object_id)
        
        # 获取物体的动力学信息（包括摩擦系数）
        try:
            dynamics_info = p.getDynamicsInfo(object_id, -1)
            
            # 安全地提取动力学信息，确保是数值类型
            object_mass = dynamics_info[0]
            if isinstance(object_mass, (int, float, np.number)):
                object_mass = float(object_mass)
            else:
                object_mass = float(np.asscalar(object_mass)) if hasattr(np, 'asscalar') else float(object_mass[0]) if isinstance(object_mass, (list, tuple, np.ndarray)) else 0.0
            
            lateral_friction = dynamics_info[1]
            if isinstance(lateral_friction, (int, float, np.number)):
                lateral_friction = float(lateral_friction)
            else:
                lateral_friction = float(lateral_friction[0]) if isinstance(lateral_friction, (list, tuple, np.ndarray)) else 0.0
            
            spinning_friction = dynamics_info[3] if len(dynamics_info) > 3 else 0.0
            if isinstance(spinning_friction, (int, float, np.number)):
                spinning_friction = float(spinning_friction)
            else:
                spinning_friction = float(spinning_friction[0]) if isinstance(spinning_friction, (list, tuple, np.ndarray)) else 0.0
            
            rolling_friction = dynamics_info[4] if len(dynamics_info) > 4 else 0.0
            if isinstance(rolling_friction, (int, float, np.number)):
                rolling_friction = float(rolling_friction)
            else:
                rolling_friction = float(rolling_friction[0]) if isinstance(rolling_friction, (list, tuple, np.ndarray)) else 0.0
            
            self.logger.info(
                "物体动力学参数 - 质量: %.3f, 侧向摩擦系数: %.3f, 自旋摩擦系数: %.4f, 滚动摩擦系数: %.4f",
                object_mass, lateral_friction, spinning_friction, rolling_friction
            )
        except Exception as e:
            self.logger.warning("无法获取物体动力学参数: %s", str(e))

        self.logger.info(
            "抓取完成 - 接触点数量: %d, 总正压力: %.4f, 侧向摩擦力: %.4f, 旋转摩擦力: %.4f, 平均摩擦系数: %.3f",
            contact_state["contact_count"],
            contact_state["total_normal_force"],
            contact_state["total_lateral_friction"],
            contact_state["total_spinning_friction"],
            contact_state["avg_friction_coefficient"]
        )

        return {
            "final_pos": close_position,
            "hold_force": fixed_force,
            "contact_state": contact_state
        }