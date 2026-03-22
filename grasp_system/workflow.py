import pybullet as p
from typing import Optional

from .config import AppConfig
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

    def run(self, gui: bool = True):
        self.runtime.connect(gui=gui)
        try:
            self.logger.info("开始执行抓取任务")
            self.runtime.step(self.config.motion.settle_steps)

            obj_pos = self.vision.locate_object(self.runtime.obj_id)
            grasp_orn = p.getQuaternionFromEuler(self.config.motion.default_orientation_euler)

            pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + self.config.motion.pre_grasp_offset]
            self.gripper.open_gripper()
            self.arm.move_to_position(pre_grasp_pos, target_orn=grasp_orn)

            grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + self.config.motion.grasp_offset]
            self.logger.info("机械臂下降到抓取位姿 %s", grasp_pos)
            self.arm.move_to_position(grasp_pos, target_orn=grasp_orn)

            # 使用固定力抓取方法代替原来的自适应抓取方法
            grip_state = self.gripper.close_with_fixed_force()

            # 直接使用固定力保持抓取
            self.runtime.step(self.config.motion.hold_steps)

            lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + self.config.motion.lift_offset]
            self.logger.info("机械臂抬升目标 %s", lift_pos)
            self.arm.move_to_position(lift_pos, target_orn=grasp_orn)
            self.logger.info("抓取任务完成")
        finally:
            self.runtime.disconnect()