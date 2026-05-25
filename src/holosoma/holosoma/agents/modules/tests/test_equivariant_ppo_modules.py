"""Tests for equivariant PPO modules.

Verifies:
1. LinearBlock forward pass and irreps_out shape.
2. EquivariantPPOActor full pipeline: act, act_inference, log_prob, entropy.
3. InvariantPPOCritic forward pass.
4. SO(3) equivariance of the actor mean and invariance of the critic value
   under a random rotation.
5. Isotropic std structure: vector action dims share the same std.
6. setup_ppo_actor_module / setup_ppo_critic_module dispatch with type "Equivariant".
"""

from __future__ import annotations

import pytest
import torch
from e3nn import o3
from holosoma.agents.modules.equivariant import (
    EquivariantFlatExtractor,
    EquivariantPPOActor,
    InvariantPPOCritic,
    LinearBlock,
)
from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_ppo_critic_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig

# ---------------------------------------------------------------------------
# Minimal test extractor
# ---------------------------------------------------------------------------
# Observation layout: [goal_displacement(3), ee_vel(3), scalar(4)]  →  irreps "2x1o + 4x0e"
# Total obs_dim = 10.

OBS_DIM = 10
ACTION_IRREPS_STR = "1x1o + 1x0e"  # 3-D vector + 1 scalar  →  action_dim = 4


class _TestExtractor(EquivariantFlatExtractor):
    irreps_out = o3.Irreps("2x1o + 4x0e")

    def __init__(self, obs_dim: int) -> None:
        super().__init__(obs_dim)

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        # vectors first, then scalars — required convention
        vec0 = flat_obs[..., 0:3]  # goal displacement (1o)
        vec1 = flat_obs[..., 3:6]  # end-effector vel  (1o)
        scs = flat_obs[..., 6:10]  # scalar features   (0e)
        return torch.cat([vec0, vec1, scs], dim=-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return _TestExtractor(obs_dim=OBS_DIM)


@pytest.fixture
def actor(extractor):
    return EquivariantPPOActor(
        extractor=extractor,
        action_irreps_str=ACTION_IRREPS_STR,
        n_scalars=8,
        n_vectors=4,
        init_noise_std=0.5,
    )


@pytest.fixture
def critic(extractor):
    return InvariantPPOCritic(
        extractor=extractor,
        hidden_dims=(32, 32),
    )


# ---------------------------------------------------------------------------
# LinearBlock
# ---------------------------------------------------------------------------


class TestLinearBlock:
    def test_output_shape(self):
        irreps_in = o3.Irreps("2x1o + 4x0e")
        block = LinearBlock(irreps_in, n_scalars=8, n_vectors=4)
        x = torch.randn(16, irreps_in.dim)
        out = block(x)
        assert out.shape == (16, block.irreps_out.dim)

    def test_irreps_out_structure(self):
        block = LinearBlock(o3.Irreps("2x1o + 4x0e"), n_scalars=8, n_vectors=4)
        # irreps_out must contain exactly 8 scalars and 4 vectors
        n_sc = sum(mul for mul, ir in block.irreps_out if ir.l == 0)
        n_ve = sum(mul for mul, ir in block.irreps_out if ir.l == 1)
        assert n_sc == 8
        assert n_ve == 4


# ---------------------------------------------------------------------------
# EquivariantPPOActor
# ---------------------------------------------------------------------------


class TestEquivariantPPOActor:
    def test_act_shape(self, actor):
        obs = torch.randn(32, OBS_DIM)
        actions = actor.act({"actor_obs": obs})
        assert actions.shape == (32, o3.Irreps(ACTION_IRREPS_STR).dim)

    def test_act_inference_shape(self, actor):
        obs = torch.randn(32, OBS_DIM)
        mean = actor.act_inference({"actor_obs": obs})
        assert mean.shape == (32, o3.Irreps(ACTION_IRREPS_STR).dim)

    def test_log_prob_shape(self, actor):
        obs = torch.randn(32, OBS_DIM)
        actions = actor.act({"actor_obs": obs})
        lp = actor.get_actions_log_prob(actions)
        assert lp.shape == (32,)

    def test_entropy_shape(self, actor):
        actor.act({"actor_obs": torch.randn(32, OBS_DIM)})
        assert actor.entropy.shape == (32,)

    def test_action_mean_std_shape(self, actor):
        actor.act({"actor_obs": torch.randn(16, OBS_DIM)})
        assert actor.action_mean.shape == (16, o3.Irreps(ACTION_IRREPS_STR).dim)
        assert actor.action_std.shape == (16, o3.Irreps(ACTION_IRREPS_STR).dim)

    def test_isotropic_std_for_vector_action(self, actor):
        """The std for all three dims of a vector action slot must be equal."""
        actor.act({"actor_obs": torch.randn(4, OBS_DIM)})
        std = actor.action_std[0]  # (action_dim,)
        # ACTION_IRREPS_STR = "1x1o + 1x0e" → dims 0,1,2 are the vector; dim 3 is scalar
        assert torch.allclose(std[0], std[1]) and torch.allclose(std[1], std[2]), (
            "Vector action dims must share the same std (isotropic Gaussian)"
        )
        # Scalar std may differ from vector std
        # (no assertion on that, just verify the shape is correct)

    def test_so3_equivariance_actor_mean(self, actor):
        """Rotating the scene by R should rotate the vector action mean by R.

        For ACTION_IRREPS_STR = "1x1o + 1x0e":
          * dims 0-2 are the vector action   → must transform as R @ mu_xyz
          * dim  3   is the scalar action    → must be unchanged
        """
        actor.eval()
        torch.manual_seed(0)
        obs = torch.randn(1, OBS_DIM)

        # Random rotation matrix via QR decomposition
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        if torch.det(R) < 0:
            R[:, 0] *= -1

        # Rotate the two vector channels in the observation
        obs_rot = obs.clone()
        obs_rot[..., 0:3] = obs[..., 0:3] @ R.T  # goal displacement
        obs_rot[..., 3:6] = obs[..., 3:6] @ R.T  # ee velocity

        with torch.no_grad():
            mu = actor.act_inference({"actor_obs": obs})
            mu_rot = actor.act_inference({"actor_obs": obs_rot})

        # Vector part of action should rotate with R
        expected_xyz = (R @ mu[..., :3].T).T
        assert torch.allclose(mu_rot[..., :3], expected_xyz, atol=1e-5), (
            "Vector action mean must be equivariant under SO(3)"
        )

        # Scalar part must be invariant
        assert torch.allclose(mu_rot[..., 3:], mu[..., 3:], atol=1e-5), (
            "Scalar action mean must be invariant under SO(3)"
        )

    def test_reset_no_error(self, actor):
        actor.reset()
        actor.reset(dones=torch.tensor([True, False]))


# ---------------------------------------------------------------------------
# InvariantPPOCritic
# ---------------------------------------------------------------------------


class TestInvariantPPOCritic:
    def test_evaluate_shape(self, critic):
        obs = torch.randn(32, OBS_DIM)
        value = critic.evaluate({"critic_obs": obs})
        assert value.shape == (32, 1)

    def test_so3_invariance_critic_value(self, critic):
        """The critic value must be identical for rotated observations."""
        critic.eval()
        torch.manual_seed(1)
        obs = torch.randn(1, OBS_DIM)

        R = torch.linalg.qr(torch.randn(3, 3))[0]
        if torch.det(R) < 0:
            R[:, 0] *= -1

        obs_rot = obs.clone()
        obs_rot[..., 0:3] = obs[..., 0:3] @ R.T
        obs_rot[..., 3:6] = obs[..., 3:6] @ R.T

        with torch.no_grad():
            v = critic.evaluate({"critic_obs": obs})
            v_rot = critic.evaluate({"critic_obs": obs_rot})

        assert torch.allclose(v, v_rot, atol=1e-5), "Critic value must be invariant under SO(3)"


# ---------------------------------------------------------------------------
# setup_ppo_*_module dispatch
# ---------------------------------------------------------------------------


class TestModuleUtilsDispatch:
    @pytest.fixture
    def equivariant_layer_cfg(self):
        return LayerConfig(
            equivariant_extractor_class=("holosoma.agents.modules.tests.test_equivariant_ppo_modules._TestExtractor"),
            equivariant_action_irreps=ACTION_IRREPS_STR,
            equivariant_n_scalars=8,
            equivariant_n_vectors=4,
            equivariant_critic_hidden_dims=(32, 32),
        )

    @pytest.fixture
    def actor_module_config(self, equivariant_layer_cfg):
        return ModuleConfig(
            type="Equivariant",
            input_dim=["actor_obs"],
            output_dim=[o3.Irreps(ACTION_IRREPS_STR).dim],
            layer_config=equivariant_layer_cfg,
        )

    @pytest.fixture
    def critic_module_config(self, equivariant_layer_cfg):
        return ModuleConfig(
            type="Equivariant",
            input_dim=["critic_obs"],
            output_dim=[1],
            layer_config=equivariant_layer_cfg,
        )

    def test_setup_actor(self, actor_module_config):
        obs_dim_dict = {"actor_obs": OBS_DIM}
        actor = setup_ppo_actor_module(
            obs_dim_dict=obs_dim_dict,
            module_config=actor_module_config,
            num_actions=o3.Irreps(ACTION_IRREPS_STR).dim,
            init_noise_std=0.5,
            device="cpu",
            history_length={"actor_obs": 1},
        )
        assert isinstance(actor, EquivariantPPOActor)
        obs = torch.randn(8, OBS_DIM)
        actions = actor.act({"actor_obs": obs})
        assert actions.shape == (8, o3.Irreps(ACTION_IRREPS_STR).dim)

    def test_setup_critic(self, critic_module_config):
        obs_dim_dict = {"critic_obs": OBS_DIM}
        critic = setup_ppo_critic_module(
            obs_dim_dict=obs_dim_dict,
            module_config=critic_module_config,
            device="cpu",
            history_length={"critic_obs": 1},
        )
        assert isinstance(critic, InvariantPPOCritic)
        obs = torch.randn(8, OBS_DIM)
        value = critic.evaluate({"critic_obs": obs})
        assert value.shape == (8, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
