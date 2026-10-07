"""RSL-RL PPO configuration for bimanual bowl pivoting: the grasp-lift settings, longer."""

from isaaclab.utils import configclass

from ...ycb_grasp.agents.rsl_rl_ppo_cfg import GraspLiftPPORunnerCfg


@configclass
class BowlPivotPPORunnerCfg(GraspLiftPPORunnerCfg):
    max_iterations = 3000
    # Logs go to outputs/rl/runs/<experiment_name>/.
    experiment_name = "bowl_pivot_024_bowl"
