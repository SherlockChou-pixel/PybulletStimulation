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

    def move_to_position(self, target_pos, target_orn=None, steps=None, grip_adjust=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler(self.config.motion.default_orientation_euler)
        if steps is None:
            steps = self.config.motion.approach_steps

        joint_poses = p.calculateInverseKinematics(
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

        for joint_id in self.arm_joints:
            p.setJointMotorControl2(
                bodyIndex=self.runtime.panda_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
                force=self.config.motion.arm_force,
                maxVelocity=self.config.motion.arm_max_velocity,
                positionGain=self.config.motion.arm_position_gain,
                velocityGain=self.config.motion.arm_velocity_gain,
            )

        self.logger.info("机械臂移动 target_pos=%s steps=%s", target_pos, steps)
        self.runtime.step(steps, hook=grip_adjust)
