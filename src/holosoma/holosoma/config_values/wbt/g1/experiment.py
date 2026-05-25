from dataclasses import replace

from holosoma.config_types.algo import LayerConfig, ModuleConfig, PPOModuleDictConfig
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_equivariant",
]

# ──────────────────────────────────────────────────────────────────────────────
# Equivariant PPO variant
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACTOR_CLASS = "holosoma.config_values.wbt.g1.equivariant_extractor.G1WBTActorExtractor"

# Actor obs dim = 154  (see equivariant_extractor.py for the full breakdown)
# Action irreps = "29x0e" — 29 scalar joint positions, invariant under rotation
#
# Hidden architecture:
#   block1: LinearBlock("3x1o + 151x0e", n_scalars=128, n_vectors=16)
#            → output "128x0e + 16x1o"  (dim = 176)
#   block2: LinearBlock("128x0e + 16x1o", n_scalars=128, n_vectors=16)
#            → output "128x0e + 16x1o"  (dim = 176)
#   mu_layer: o3.Linear("128x0e + 16x1o", "29x0e")  → 29 scalars
#
# The 16 hidden vector channels carry equivariant orientation-error
# representations through the network even though the final output is
# purely scalar.  The Gate nonlinearity uses the augmented invariants
# (norms + dot products of the 3 input vectors) to gate these channels.
#
# Critic: unchanged standard MLP [512, 256, 128] on the full 286-d critic_obs.
# Using a plain invariant MLP for the critic provides stable training as
# observed in the SAC reference implementation.
#
# Required PPO settings:
#   empirical_normalization = False   (mean-subtraction would break equivariance)
#   init_noise_std = 1.0              (same as base experiment)

g1_29dof_wbt_equivariant = replace(
    g1_29dof_wbt,
    algo=replace(
        g1_29dof_wbt.algo,
        config=replace(
            g1_29dof_wbt.algo.config,
            # Equivariance requires raw observations — disable mean/var normalization.
            empirical_normalization=False,
            module_dict=PPOModuleDictConfig(
                actor=ModuleConfig(
                    type="Equivariant",
                    input_dim=["actor_obs"],
                    output_dim=[29],  # 29 joint positions
                    layer_config=LayerConfig(
                        # Extractor: parses flat actor_obs into irreps
                        equivariant_extractor_class=_EXTRACTOR_CLASS,
                        # Action space: 29 scalar (0e) joint positions
                        # Invariant under any rotation of the base frame.
                        equivariant_action_irreps="29x0e",
                        # Hidden layer width:
                        #   128 scalars  — learns nonlinear invariant features
                        #   16 vectors   — maintains equivariant orientation-error
                        #                  representations between layers
                        equivariant_n_scalars=128,
                        equivariant_n_vectors=16,
                    ),
                ),
                critic=ModuleConfig(
                    type="MLP",
                    input_dim=["critic_obs"],
                    output_dim=[1],
                    layer_config=LayerConfig(
                        hidden_dims=[512, 256, 128],
                        activation="ELU",
                    ),
                ),
            ),
        ),
    ),
)

"""
Train with the equivariant actor:

python src/holosoma/holosoma/train_agent.py \\
    exp:g1-29dof-wbt-equivariant \\
    logger:wandb \\
    --command.setup_terms.motion_command.params.motion_config.motion_file="<PATH>.npz"

Vs the MLP baseline:

python src/holosoma/holosoma/train_agent.py \\
    exp:g1-29dof-wbt \\
    logger:wandb \\
    --command.setup_terms.motion_command.params.motion_config.motion_file="<PATH>.npz"

Expected benefits over the MLP baseline:
  * Faster convergence on motions with varied initial yaw headings.
  * Better generalisation when yaw noise in init_pose_config is high.
  * No sensitivity to the robot's absolute facing direction.
"""

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
