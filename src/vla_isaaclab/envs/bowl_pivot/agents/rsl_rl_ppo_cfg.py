"""RSL-RL PPO configuration for bimanual bowl pivoting: the grasp-lift settings, longer."""

from isaaclab.utils import configclass

from ...ycb_grasp.agents.rsl_rl_ppo_cfg import GraspLiftPPORunnerCfg


@configclass
class BowlPivotPPORunnerCfg(GraspLiftPPORunnerCfg):
    max_iterations = 3000
    # Logs go to outputs/rl/runs/<experiment_name>/.
    experiment_name = "bowl_pivot_024_bowl"


@configclass
class BowlPivotLiftPPORunnerCfg(BowlPivotPPORunnerCfg):
    """Pivot-then-lift: less entropy; at 0.01 the action noise kept growing (1.0 -> 1.56)
    while the policy never learned the second stage, and at 0.003 it still grew (to 1.9)."""

    algorithm = BowlPivotPPORunnerCfg().algorithm.replace(entropy_coef=0.001)


@configclass
class BowlFlippedLiftPPORunnerCfg(GraspLiftPPORunnerCfg):
    max_iterations = 200
    experiment_name = "bowl_flipped_lift_024_bowl"


@configclass
class BowlFlippedLiftRightPPORunnerCfg(GraspLiftPPORunnerCfg):
    max_iterations = 300
    experiment_name = "bowl_flipped_lift_right_024_bowl"
