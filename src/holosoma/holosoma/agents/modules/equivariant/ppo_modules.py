"""Equivariant PPO actor and invariant PPO critic built on e3nn.

Architecture overview
---------------------

Actor  (SO(3)-equivariant)
~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    flat_obs  ──►  EquivariantFlatExtractor  ──►  irrep tensor  x
                ──►  _augment(x)                              x_aug
                ──►  LinearBlock  ──►  LinearBlock            h
                ──►  o3.Linear                                mean   (action_irreps)
                ──►  _build_std()                             std    (action_irreps, isotropic per vector slot)
                ──►  Normal(mean, std)

Key equivariance properties:

* The extractor encodes the scene in SO(3)-covariant irreps.
* ``_augment`` adds rotation-invariant scalars (norms + dot products of all
  ``1o`` channels) so that the Gate nonlinearity's gate scalars have access to
  geometric information — fixing the "blind Gate" problem from Schur's lemma.
* The mean action is produced by ``o3.Linear``, which maps
  ``n_scalars x 0e → 0e`` and ``n_vectors x 1o → 1o`` (never 1o→0e),
  so the vector action components rotate with the scene.
* The std for each vector action slot is a *single* shared scalar so the noise
  distribution ``N(Rμ, σ²I) = R·N(μ, σ²I)`` stays isotropic and equivariant.

Critic  (SO(3)-invariant)
~~~~~~~~~~~~~~~~~~~~~~~~~
::

    flat_obs  ──►  EquivariantFlatExtractor  ──►  irrep tensor  x
                ──►  _invariant_features(x)                   scalars
                ──►  MLP  ──►  scalar value

Invariant features extracted systematically from the irreps output:
  * ‖vᵢ‖  for each vector channel  vᵢ  (norms are rotation-invariant)
  * vᵢ · vⱼ  for all i < j           (dot products are rotation-invariant)
  * all scalar (0e) channels pass through unchanged

Using an invariant MLP critic (rather than a Tensor-Product critic) provides
numerically stable training, as observed in the SAC reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from torch.distributions import Normal

from .extractor import EquivariantFlatExtractor
from .linear_block import LinearBlock

# ---------------------------------------------------------------------------
# Helpers: scalar augmentation and invariant feature extraction
# ---------------------------------------------------------------------------


def _aug_irreps(irreps_out: o3.Irreps) -> o3.Irreps:
    """Augmented input irreps: original vectors + original scalars + norms + pairwise dots.

    o3.Linear respects Schur's lemma: it can never produce a scalar (0e) from a
    vector (1o).  This means Gate scalars — which multiplicatively gate vector
    channels — are blind to vector norms and alignments unless we explicitly
    append those invariants as extra 0e channels before the first LinearBlock.

    The augmented irreps are: ``n_vec x 1o + (n_sc + n_aug) x 0e`` where
    ``n_aug = n_vec + C(n_vec, 2)``  (norms + pairwise dot products).
    """
    n_vec = sum(mul for mul, ir in irreps_out if ir.l == 1 and ir.p == -1)
    n_sc = sum(mul for mul, ir in irreps_out if ir.l == 0)
    n_aug = n_vec + n_vec * (n_vec - 1) // 2  # norms + C(n_vec, 2) dots
    return o3.Irreps(f"{n_vec}x1o + {n_sc + n_aug}x0e")


def _augment(x: torch.Tensor, irreps_out: o3.Irreps) -> torch.Tensor:
    """Append norms and pairwise dot products of all 1o channels to x.

    The output has the same 1o channels followed by the original 0e channels
    plus the new invariant scalars — matching ``_aug_irreps(irreps_out)``.
    """
    vecs: list[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps_out:
        for m in range(mul):
            chunk = x[..., offset + m * ir.dim : offset + (m + 1) * ir.dim]
            if ir.l == 1 and ir.p == -1:
                vecs.append(chunk)
        offset += mul * ir.dim

    extras: list[torch.Tensor] = [x]
    for v in vecs:
        extras.append(v.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            extras.append((vecs[i] * vecs[j]).sum(dim=-1, keepdim=True))
    return torch.cat(extras, dim=-1)


def _n_inv_features(irreps_out: o3.Irreps) -> int:
    """Number of rotation-invariant scalars extractable from an irreps tensor."""
    n_vec = sum(mul for mul, ir in irreps_out if ir.l == 1 and ir.p == -1)
    n_sc = sum(mul for mul, ir in irreps_out if ir.l == 0)
    return n_vec + n_vec * (n_vec - 1) // 2 + n_sc


def _invariant_features(x: torch.Tensor, irreps_out: o3.Irreps) -> torch.Tensor:
    """Extract rotation-invariant scalar features from an irrep-structured tensor.

    Returns a flat tensor containing:
    * ``‖vᵢ‖`` for every vector channel ``vᵢ``
    * ``vᵢ · vⱼ`` for all ``i < j``
    * all scalar (0e) channels unchanged
    """
    vecs: list[torch.Tensor] = []
    scalars: list[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps_out:
        for m in range(mul):
            chunk = x[..., offset + m * ir.dim : offset + (m + 1) * ir.dim]
            if ir.l == 1 and ir.p == -1:
                vecs.append(chunk)
            elif ir.l == 0:
                scalars.append(chunk)
        offset += mul * ir.dim

    feats: list[torch.Tensor] = []
    for v in vecs:
        feats.append(v.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            feats.append((vecs[i] * vecs[j]).sum(dim=-1, keepdim=True))
    feats.extend(scalars)
    return torch.cat(feats, dim=-1)


def _build_std_expand_idx(action_irreps: o3.Irreps) -> torch.Tensor:
    """Build an index tensor to expand per-slot std parameters to full action dim.

    For ``action_irreps = "1x1o + 2x0e"`` (action_dim = 5):
    * slot 0 (``1o``, dim=3): std param index 0 → indices [0, 0, 0]
    * slot 1 (``0e``, dim=1): std param index 1 → index  [1]
    * slot 2 (``0e``, dim=1): std param index 2 → index  [2]
    ⟹ ``expand_idx = [0, 0, 0, 1, 2]``,  ``n_std_params = 3``

    This ensures the three spatial dims of a vector action always share the
    same std (isotropic Gaussian), which is the only noise distribution that
    commutes with SO(3) rotations: ``N(Rμ, σ²I) = R·N(μ, σ²I)``.
    """
    idx: list[int] = []
    param_idx = 0
    for mul, ir in action_irreps:
        for _ in range(mul):
            idx.extend([param_idx] * ir.dim)
            param_idx += 1
    return torch.tensor(idx, dtype=torch.long)


# ---------------------------------------------------------------------------
# Equivariant actor
# ---------------------------------------------------------------------------


class EquivariantPPOActor(nn.Module):
    """SO(3)-equivariant stochastic actor for PPO.

    Drop-in replacement for :class:`~holosoma.agents.modules.ppo_modules.PPOActor`.
    Implements the same interface used by :class:`~holosoma.agents.ppo.ppo.PPO`:
    ``act``, ``act_inference``, ``update_distribution``, ``get_actions_log_prob``,
    ``action_mean``, ``action_std``, ``entropy``, and ``reset``.

    Parameters
    ----------
    extractor:
        A task-specific :class:`EquivariantFlatExtractor` subclass that converts
        the flat observation vector into an irrep-structured tensor.
    action_irreps_str:
        e3nn irreps string describing the action structure, e.g.
        ``"1x1o + 2x0e"`` for a 3-D displacement vector + 2 scalar actions.
        Vector slots (``1o``) use a single shared std per slot (isotropic);
        scalar slots (``0e``) each get their own std parameter.
    n_scalars:
        Number of scalar (0e) channels in each hidden ``LinearBlock``.
    n_vectors:
        Number of vector (1o) channels in each hidden ``LinearBlock``.
    init_noise_std:
        Initial value for all std parameters (log-space initialisation).

    Notes
    -----
    * Set ``empirical_normalization: false`` in the PPO config — mean-subtracting
      vector channels breaks equivariance.
    * Two hidden ``LinearBlock`` layers are used (matching the SAC reference).
      Add more by extending the constructor if the task requires deeper nets.
    """

    def __init__(
        self,
        extractor: EquivariantFlatExtractor,
        action_irreps_str: str,
        n_scalars: int = 16,
        n_vectors: int = 8,
        init_noise_std: float = 0.8,
    ) -> None:
        super().__init__()

        self.extractor = extractor
        self.action_irreps = o3.Irreps(action_irreps_str)

        # ── Equivariant backbone ──────────────────────────────────────────
        aug = _aug_irreps(extractor.irreps_out)
        self.block1 = LinearBlock(aug, n_scalars, n_vectors)
        self.block2 = LinearBlock(self.block1.irreps_out, n_scalars, n_vectors)

        # o3.Linear: maps n_scalars x 0e → 0e and n_vectors x 1o → 1o only
        self.mu_layer = o3.Linear(self.block2.irreps_out, self.action_irreps)

        # ── Structured std parameter ──────────────────────────────────────
        # One log-std per irrep *slot* (not per dimension): vector slots share
        # a single std so the noise remains isotropic for SO(3) equivariance.
        n_std_params = sum(mul for mul, _ in self.action_irreps)
        self.std_log = nn.Parameter(torch.log(torch.ones(n_std_params) * init_noise_std))

        # Pre-compute the index that expands (n_std_params,) → (action_dim,)
        self.register_buffer("_std_expand_idx", _build_std_expand_idx(self.action_irreps))

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        action_dim = self.action_irreps.dim
        print(
            f"EquivariantPPOActor | extractor irreps: {extractor.irreps_out} "
            f"| action irreps: {self.action_irreps} (dim={action_dim}) "
            f"| hidden: {n_scalars}s+{n_vectors}v | std params: {n_std_params}"
        )

    # ── Std helpers ───────────────────────────────────────────────────────

    def _build_std(self) -> torch.Tensor:
        """Expand per-slot log-stds to a ``(action_dim,)`` std tensor."""
        return self.std_log.exp()[self._std_expand_idx]

    # ── Scalar augmentation (same logic as the SAC reference) ─────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        return _augment(x, self.extractor.irreps_out)

    # ── Core forward ─────────────────────────────────────────────────────

    def update_distribution(self, actor_obs: torch.Tensor) -> None:
        """Compute mean and build the stochastic action distribution.

        Called by :meth:`act` during rollout and by :meth:`_compute_ppo_loss`
        during training.
        """
        x = self.extractor(actor_obs)  # (..., irreps_out.dim)
        x_aug = self._augment(x)  # (..., aug_irreps.dim)
        h = self.block2(self.block1(x_aug))  # (..., hidden_dim)
        mean = self.mu_layer(h)  # (..., action_dim)
        std = self._build_std()  # (action_dim,)
        self.distribution = Normal(mean, mean * 0.0 + std)

    # ── PPOActor interface ────────────────────────────────────────────────

    def act(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.update_distribution(policy_state_dict["actor_obs"])
        return self.distribution.sample()

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic mean action — used for symmetry loss and ONNX export."""
        actor_obs = policy_state_dict["actor_obs"]
        x = self.extractor(actor_obs)
        x_aug = self._augment(x)
        h = self.block2(self.block1(x_aug))
        return self.mu_layer(h)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Sum log-prob across action dimensions for a batch of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None) -> None:
        """No hidden state to reset for an MLP-based actor."""

    # ── Distribution properties ────────────────────────────────────────────

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)


# ---------------------------------------------------------------------------
# Invariant critic
# ---------------------------------------------------------------------------


class InvariantPPOCritic(nn.Module):
    """SO(3)-invariant value function for PPO.

    Drop-in replacement for :class:`~holosoma.agents.modules.ppo_modules.PPOCritic`.
    Computes a scalar value estimate from the flat observation by:

    1. Passing through the equivariant extractor to obtain irrep-structured features.
    2. Reducing to rotation-invariant scalars (vector norms + pairwise dot products +
       original scalar channels).
    3. Feeding those scalars through a plain MLP → scalar value.

    Using an invariant MLP (rather than a Tensor-Product critic) avoids the
    unstable early training observed with TP critics in the SAC reference
    (loss_q spikes, exploding alpha) because the simpler landscape converges
    quickly.

    Parameters
    ----------
    extractor:
        A task-specific :class:`EquivariantFlatExtractor` subclass.  Can be the
        same class as the actor's extractor (both are instantiated separately).
    hidden_dims:
        Hidden layer sizes for the invariant MLP.
    activation:
        Activation class (default: ``nn.ELU``).
    """

    def __init__(
        self,
        extractor: EquivariantFlatExtractor,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        self.extractor = extractor
        n_features = _n_inv_features(extractor.irreps_out)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        print(
            f"InvariantPPOCritic | extractor irreps: {extractor.irreps_out} "
            f"| invariant features: {n_features} | hidden: {list(hidden_dims)}"
        )

    def _invariant_features(self, flat_obs: torch.Tensor) -> torch.Tensor:
        x = self.extractor(flat_obs)
        return _invariant_features(x, self.extractor.irreps_out)

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate of shape ``(B, 1)``."""
        features = self._invariant_features(policy_state_dict["critic_obs"])
        return self.mlp(features)

    def reset(self, dones=None) -> None:
        """No hidden state to reset."""

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states) -> None:
        pass
