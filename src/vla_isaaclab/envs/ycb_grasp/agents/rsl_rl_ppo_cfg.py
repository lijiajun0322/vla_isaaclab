"""RSL-RL PPO configuration for YCB grasp-and-lift."""

from isaaclab.utils import configclass

import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class GraspLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "grasp_lift"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        # Explore widely: a smaller start noise collapsed to "keep still" by iteration 100.
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# -- one config per object (logs go to outputs/rl/runs/<experiment_name>/) --------


@configclass
class SugarBoxGraspLiftPPORunnerCfg(GraspLiftPPORunnerCfg):
    experiment_name = "grasp_lift_004_sugar_box"


@configclass
class MustardBottleGraspLiftPPORunnerCfg(GraspLiftPPORunnerCfg):
    experiment_name = "grasp_lift_006_mustard_bottle"
