# equivariant_reach_extractor.py

from typing import Optional

import gymnasium.spaces as spaces
import torch
from e3nn import o3

from .base_extractor import BaseExtractor


class EquivariantReachExtractor(BaseExtractor):
    """
    Converts a FetchReach / PandaReach GoalEnv dict observation into
    equivariant irrep features, encoding translation invariance by
    construction instead of learning it from data.

    Raw dict keys used:
        observation  (10,): [ee_pos(3), gripper_state(2), ee_vel(3), gripper_vel(2)]
        achieved_goal (3,): current end-effector position
        desired_goal  (3,): target end-effector position

    Output tensor layout  →  irreps "2x1o + 4x0e"  (dim = 10):
        dims  0-2  : error_vec  = desired_goal − achieved_goal   [1x1o  polar vector]
        dims  3-5  : ee_vel     = observation[5:8]               [1x1o  polar vector]
        dims  6-7  : gripper finger positions = observation[3:5] [2x0e  scalars]
        dims  8-9  : gripper finger velocities = observation[8:10][2x0e scalars]

    Translation invariance: absolute positions are never used; only the
    goal-relative displacement appears in the features.
    """

    irreps_out: o3.Irreps = o3.Irreps("2x1o + 4x0e")  # type: ignore[assignment]

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space=observation_space)
        self.n_features = self.irreps_out.dim  # 10

    def forward(
        self,
        observation: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        achieved = torch.as_tensor(observation["achieved_goal"], dtype=torch.float32, device=device)
        desired = torch.as_tensor(observation["desired_goal"], dtype=torch.float32, device=device)
        raw = torch.as_tensor(observation["observation"], dtype=torch.float32, device=device)

        # Translation-invariant goal error  (1x1o)
        error_vec = desired - achieved  # (..., 3)
        # End-effector velocity               (1x1o)
        ee_vel = raw[..., 5:8]  # (..., 3)
        # Gripper scalars                     (4x0e)
        gripper_sc = torch.cat([raw[..., 3:5], raw[..., 8:10]], dim=-1)  # (..., 4)

        # Layout must match irreps "2x1o + 4x0e":
        #   vectors (1o) before scalars (0e)
        return torch.cat([error_vec, ee_vel, gripper_sc], dim=-1)  # (..., 10)


# equivariant_policy.py

"""
Full nonlinear equivariant SAC policy for goal-reaching tasks (Reach).

Key design decisions vs the MLP baseline
─────────────────────────────────────────
Actor
  • Two LinearBlock hidden layers keep the actor equivariant under SO(3):
    rotating the scene rotates the output action by the same amount.
  • Mean output is "1x1o + 1x0e" (displacement vector + gripper scalar).
  • Log-std output is "2x0e" — two *invariant* scalars:
      log_std[0]  shared isotropic std for the xyz spatial action
      log_std[1]  std for the gripper scalar action
    Using a *single* std for all three spatial dims is the only choice that
    keeps the Gaussian distribution equivariant: N(Rμ, σ²I) = R·N(μ, σ²I).
  • xyz squashing uses norm_squash (scale magnitude, preserve direction)
    instead of element-wise tanh, which would break equivariance.

Why we augment the scalar input channels
─────────────────────────────────────────
  o3.Linear can only map 0e→0e and 1o→1o (Schur's lemma: zero weights are
  allocated for any 1o→0e path).  This means the Gate scalars — which
  multiplicatively control the vector channels — can only be functions of
  the *original scalar inputs* (gripper_sc).  After two LinearBlocks the
  actor's effective form is:

      mu_xyz  = f(gripper_sc)·error_vec + g(gripper_sc)·ee_vel
      log_std = h(gripper_sc)

  where f, g, h are constants (gripper barely moves in a reach task).
  This makes the actor a fixed-gain PD controller that cannot reduce its
  proportional gain or noise as it nears the goal → oscillation.

  Fix: pre-compute all pairwise SO(3)-invariant scalars (norms + dot
  products) from the vector channels and append them as extra 0e features
  before block1.  For an extractor with N vector channels this adds
  N norms + N*(N-1)/2 dot products = N*(N+1)/2 extra scalars.

  Examples:
    Reach  2×1o + 4×0e  →  2×1o + 7×0e   (+3 scalars: 2 norms, 1 dot)
    Push   4×1o + 4×0e  →  4×1o + 14×0e  (+10 scalars: 4 norms, 6 dots)

  Equivariance is preserved because norms and dot products of vectors are
  rotation-invariant scalars (0e irreps) by construction.
  The augmentation is computed automatically in _aug_irreps / _augment.

Q-function
  • Standard MLP over hand-crafted SO(3)-invariant scalar features.
    Replacing the earlier TPBlock design: empirically the TP Q-function
    diverged badly early in training (loss_q peaked at 49) because many
    Clebsch-Gordan paths make the loss landscape hard to optimise before
    the weights settle.  An MLP critic converges quickly and stably.
  • Invariant features extracted from obs + action (11 scalars total):
      ‖error_vec‖, ‖ee_vel‖, ‖act_xyz‖          norms
      error_vec·ee_vel, error_vec·act_xyz,
      ee_vel·act_xyz                             inner products
      gripper_sc (4), act_g (1)                 already scalars
  • Two hidden layers [64 × 64] with ReLU, scalar output.
"""

import math
from typing import Union

import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from numpy.typing import NDArray

from ..extractors.base_extractor import BaseExtractor
from ..extractors.equivariant_reach_extractor import EquivariantReachExtractor
from ..models.equivariant_blocks import LinearBlock
from ..models.mlp import mlp
from ..utils.observation import unsqueeze_observation

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ── Equivariant squashing helpers ─────────────────────────────────────────────


def norm_squash(v: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Equivariant squashing for a 3-D action vector.

    Scales the magnitude by tanh and preserves the direction:
        f(v) = limit · tanh(‖v‖) · v / ‖v‖

    Equivariance proof:
        f(Rv) = limit · tanh(‖Rv‖) · Rv / ‖Rv‖
              = limit · tanh(‖v‖)  · R(v / ‖v‖)
              = R · f(v)   ✓
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return limit * torch.tanh(norm) * (v / norm)


def norm_squash_log_det(pre_squash: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Log |det Jacobian| of norm_squash for a 3-D vector.

    J has two distinct eigenvalues:
        limit · tanh(r)/r   (multiplicity 2, perpendicular to v)
        limit · sech²(r)    (multiplicity 1, along v)

    det J = limit³ · (tanh(r)/r)² · sech²(r)

    log|det J| = 3·log(limit)
               + 2·[log tanh(r) − log r]
               + log(1 − tanh²(r))

    This is well-behaved at r → 0 (limit³ · I Jacobian)
    and diverges to −∞ at r → ∞ (boundary of action space).
    """
    r = pre_squash.norm(dim=-1).clamp(min=1e-8)  # (B,)
    tanh_r = torch.tanh(r)
    return (
        3.0 * math.log(limit)
        + 2.0 * (torch.log(tanh_r.clamp(min=1e-8)) - torch.log(r))
        + torch.log(1.0 - tanh_r**2 + 1e-6)
    )


# ── Actor ─────────────────────────────────────────────────────────────────────


class EquivariantSACActor(nn.Module):
    """
    Stochastic equivariant actor for SAC.

    Interface identical to SquashedGaussianMLPActor:
        forward(obs, deterministic, with_logprob) → (action, logp)

    Architecture:
        extractor  →  LinearBlock  →  LinearBlock
                   →  mu_layer  ("1x1o + 1x0e")   equivariant mean
                   →  log_std_layer ("2x0e")        invariant log-stds
    """

    @staticmethod
    def _aug_irreps(irreps_out: o3.Irreps) -> o3.Irreps:
        """Augmented input irreps: original + norms + pairwise dot products."""
        n_vec = sum(mul for mul, ir in irreps_out if ir.l == 1 and ir.p == -1)
        n_sc = sum(mul for mul, ir in irreps_out if ir.l == 0)
        n_aug = n_vec + n_vec * (n_vec - 1) // 2  # norms + C(n_vec, 2) dots
        return o3.Irreps(f"{n_vec}x1o + {n_sc + n_aug}x0e")

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Append norms and pairwise dot products of all 1o channels to x."""
        vecs, offset = [], 0
        for mul, ir in self.extractor.irreps_out:
            for m in range(mul):
                chunk = x[..., offset + m * ir.dim : offset + (m + 1) * ir.dim]
                if ir.l == 1 and ir.p == -1:
                    vecs.append(chunk)
            offset += mul * ir.dim
        extras = [x]
        for v in vecs:
            extras.append(v.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                extras.append((vecs[i] * vecs[j]).sum(-1, keepdim=True))
        return torch.cat(extras, dim=-1)

    def __init__(
        self,
        extractor: nn.Module,
        act_limit: float,
        n_scalars: int = 16,
        n_vectors: int = 8,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.act_limit = act_limit

        self.block1 = LinearBlock(self._aug_irreps(extractor.irreps_out), n_scalars, n_vectors)
        self.block2 = LinearBlock(self.block1.irreps_out, n_scalars, n_vectors)

        hidden = self.block2.irreps_out  # "n_scalars x 0e + n_vectors x 1o"

        # Equivariant mean: displacement vector (1x1o) + gripper scalar (1x0e)
        # o3.Linear maps:  n_scalars x 0e → 1x0e  and  n_vectors x 1o → 1x1o
        self.mu_layer = o3.Linear(hidden, o3.Irreps("1x1o + 1x0e"))

        # Invariant log-stds: two scalars only (0e)
        #   index 0 → shared isotropic σ for all three xyz dims
        #   index 1 → σ for gripper scalar
        self.log_std_layer = o3.Linear(hidden, o3.Irreps("2x0e"))

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        with_logprob: bool = True,
    ):
        device = next(self.parameters()).device
        x = self.extractor(obs, device=device)

        # ── augment scalar channels with geometry-aware invariants ────────
        # o3.Linear cannot produce 0e from 1o (Schur's lemma), so without
        # this step the Gate values are blind to vector norms and alignments.
        # _augment() adds norms + pairwise dot products of all 1o channels.
        x_aug = self._augment(x)

        h = self.block2(self.block1(x_aug))  # (B, hidden_dim)

        # ── mean and log-std ──────────────────────────────────────────────
        mu = self.mu_layer(h)  # (B, 4)
        log_std = self.log_std_layer(h).clamp(  # (B, 2)
            LOG_STD_MIN, LOG_STD_MAX
        )
        std = log_std.exp()  # (B, 2)

        # mu layout follows "1x1o + 1x0e": first 3 dims = vector, last = scalar
        mu_xyz = mu[..., :3]  # (B, 3)  equivariant vector
        mu_g = mu[..., 3:4]  # (B, 1)  invariant scalar

        std_xyz = std[..., 0:1]  # (B, 1)  shared for all 3 spatial dims
        std_g = std[..., 1:2]  # (B, 1)

        # ── sample ───────────────────────────────────────────────────────
        if deterministic:
            pre_xyz = mu_xyz
            pre_g = mu_g
        else:
            # Reparameterisation trick; isotropic noise preserves equivariance
            pre_xyz = mu_xyz + std_xyz * torch.randn_like(mu_xyz)
            pre_g = mu_g + std_g * torch.randn_like(mu_g)

        # ── equivariant squashing ─────────────────────────────────────────
        action_xyz = norm_squash(pre_xyz, self.act_limit)  # (B, 3)
        action_g = self.act_limit * torch.tanh(pre_g)  # (B, 1)
        action = torch.cat([action_xyz, action_g], dim=-1)  # (B, 4)

        # ── log probability ───────────────────────────────────────────────
        logp = None
        if with_logprob:
            # --- xyz: isotropic Gaussian + norm squashing ---
            # log p(pre_xyz) = −3/2·log(2π) − 3·log(σ_xyz)
            #                  − ‖pre_xyz − μ_xyz‖² / (2·σ_xyz²)
            sq_err_xyz = ((pre_xyz - mu_xyz) ** 2).sum(-1)  # (B,)
            log_p_pre_xyz = (
                -1.5 * math.log(2.0 * math.pi)
                - 3.0 * log_std[..., 0]  # 3 × (−log σ_xyz)
                - 0.5 * sq_err_xyz / (std_xyz.squeeze(-1) ** 2 + 1e-8)
            )  # (B,)
            logp_xyz = log_p_pre_xyz - norm_squash_log_det(pre_xyz, self.act_limit)

            # --- gripper: scalar Gaussian + tanh squashing ---
            sq_err_g = ((pre_g - mu_g) / (std_g + 1e-8)).squeeze(-1) ** 2
            log_p_pre_g = -0.5 * math.log(2.0 * math.pi) - log_std[..., 1] - 0.5 * sq_err_g  # (B,)
            # Numerically stable tanh correction (same form as original SAC)
            log_det_g = torch.log(self.act_limit * (1.0 - torch.tanh(pre_g).pow(2)) + 1e-6).squeeze(-1)  # (B,)
            logp_g = log_p_pre_g - log_det_g

            logp = logp_xyz + logp_g  # (B,)

        return action, logp


# ── Q-function ────────────────────────────────────────────────────────────────


class InvariantMLPQFunction(nn.Module):
    """
    SO(3)-invariant Q-function: MLP over automatically-derived invariant scalars.

    Works with any extractor whose irreps_out contains 1o vectors and 0e scalars.
    Features are computed systematically — no task-specific hard-coding:

        For N obs vectors (1o channels) and M obs scalars (0e channels):

          N   obs-vector norms           ‖vᵢ‖
          C(N,2)  obs-obs dot products   vᵢ · vⱼ  (i < j)
          1   action-vector norm         ‖act_xyz‖
          N   obs-action dot products    vᵢ · act_xyz
          M   obs scalar passthrough
          1   gripper action scalar      act_g
          ──────────────────────────────
          N·(N+1)/2 + N + M + 2  total

        Reach  N=2, M=4  →  3 + 2 + 4 + 2 = 11 features
        Push   N=4, M=4  →  10 + 4 + 4 + 2 = 20 features

    Why not TPBlock?
    The TP critic diverged early in training (loss_q peaked at 49, alpha
    spiralled) because many CG paths make the landscape hard to optimise.
    A plain invariant MLP converges stably and gives the equivariant actor
    clean Q-gradients from the start.
    """

    @staticmethod
    def _n_features(irreps_out: o3.Irreps) -> int:
        n_vec = sum(mul for mul, ir in irreps_out if ir.l == 1 and ir.p == -1)
        n_sc = sum(mul for mul, ir in irreps_out if ir.l == 0)
        return n_vec + n_vec * (n_vec - 1) // 2 + 1 + n_vec + n_sc + 1

    def __init__(
        self,
        extractor: BaseExtractor,
        hidden_sizes: tuple = (64, 64),
        activation: type = nn.ReLU,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        n = self._n_features(extractor.irreps_out)
        self.q = mlp([n] + list(hidden_sizes) + [1], activation)

    def _invariant_features(self, obs: dict, act: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = self.extractor(obs, device=device)

        # Split extractor output into 1o vector channels and 0e scalar channels
        vecs, scalars_obs = [], []
        offset = 0
        for mul, ir in self.extractor.irreps_out:
            for m in range(mul):
                chunk = x[..., offset + m * ir.dim : offset + (m + 1) * ir.dim]
                if ir.l == 1 and ir.p == -1:
                    vecs.append(chunk)
                elif ir.l == 0:
                    scalars_obs.append(chunk)
            offset += mul * ir.dim

        va = act[..., :3]  # (B, 3)  spatial action vector
        act_g = act[..., 3:4]  # (B, 1)  gripper action scalar

        feats = []
        for v in vecs:  # obs-vector norms
            feats.append(v.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        for i in range(len(vecs)):  # obs-obs dots
            for j in range(i + 1, len(vecs)):
                feats.append((vecs[i] * vecs[j]).sum(-1, keepdim=True))
        feats.append(va.norm(dim=-1, keepdim=True))  # action norm
        for v in vecs:  # obs-action dots
            feats.append((v * va).sum(-1, keepdim=True))
        feats.extend(scalars_obs)  # obs scalars
        feats.append(act_g)  # gripper action
        return torch.cat(feats, dim=-1)

    def forward(self, obs: dict, act: torch.Tensor) -> torch.Tensor:
        feats = self._invariant_features(obs, act, act.device)
        return self.q(feats).squeeze(-1)


# ── Combined policy ───────────────────────────────────────────────────────────


class EquivariantPolicy(nn.Module):
    """
    Generic SAC-compatible equivariant policy.

    Pass any equivariant extractor class via `extractor_class`; the actor
    and Q-function are built automatically from its irreps_out.  This
    mirrors how MlpPolicy accepts `extractor_type`.

        EquivariantPolicy(obs_space, act_space)
            → uses EquivariantReachExtractor (default, backward-compatible)

        EquivariantPolicy(obs_space, act_space,
                          extractor_class=EquivariantPushExtractor)
            → uses EquivariantPushExtractor

    Interface (same as MlpPolicy):
        policy.pi(obs)              → (action, logp)
        policy.q1(obs, act)         → Q-value tensor  (B,)
        policy.q2(obs, act)         → Q-value tensor  (B,)
        policy.act(obs, det=False)  → numpy action     (act_dim,)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        extractor_type: type = EquivariantReachExtractor,
        n_scalars: int = 16,
        n_vectors: int = 8,
        critic_hidden_sizes: tuple = (64, 64),
        clip_action: bool = True,
    ) -> None:
        super().__init__()
        self.clip_action = clip_action
        self.act_limit = float(action_space.high[0])  # type: ignore[union-attr]

        extractor = extractor_type(observation_space)

        self.pi = EquivariantSACActor(extractor, self.act_limit, n_scalars, n_vectors)
        self.q1 = InvariantMLPQFunction(extractor, hidden_sizes=critic_hidden_sizes)
        self.q2 = InvariantMLPQFunction(extractor, hidden_sizes=critic_hidden_sizes)

    def act(
        self,
        obs: Union[NDArray, dict],
        deterministic: bool = False,
    ) -> NDArray:
        with torch.no_grad():
            a, _ = self.pi(
                unsqueeze_observation(obs),
                deterministic=deterministic,
                with_logprob=False,
            )
            a = a.squeeze(0).cpu().numpy()

        if self.clip_action:
            a = np.clip(a, -self.act_limit, self.act_limit)

        return a
