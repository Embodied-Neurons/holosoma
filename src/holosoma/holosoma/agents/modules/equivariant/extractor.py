"""Abstract base class for equivariant feature extractors over flat observation tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from e3nn import o3


class EquivariantFlatExtractor(nn.Module, ABC):
    """Abstract base for equivariant extractors that consume flat observation tensors.

    In holosoma, observations are concatenated flat tensors rather than dicts.
    Subclasses parse a flat tensor of shape ``(..., obs_dim)`` into an
    irrep-structured tensor of shape ``(..., irreps_out.dim)`` where
    ``irreps_out`` is a class-level attribute.

    Design conventions (required by :class:`EquivariantPPOActor`):

    * Polar vectors (``1o`` irreps) must appear **before** scalars (``0e``)
      in the output layout.  This is required by ``_augment`` so that it can
      identify which channels are vectors.
    * Build **translation-invariant** features by construction (relative
      displacements, not absolute positions).  This lets you keep
      ``empirical_normalization: false`` in the PPO config without sacrificing
      robustness.

    Example subclass
    ----------------
    Suppose the observation is ``[goal_displacement(3), ee_vel(3), scalars(N)]``::

        class MyExtractor(EquivariantFlatExtractor):
            irreps_out = o3.Irreps("2x1o + Nx0e")

            def __init__(self, obs_dim: int) -> None:
                super().__init__(obs_dim)

            def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
                vec0 = flat_obs[..., 0:3]    # goal displacement  (1o)
                vec1 = flat_obs[..., 3:6]    # end-effector vel   (1o)
                scs  = flat_obs[..., 6:]     # scalar features    (0e)
                return torch.cat([vec0, vec1, scs], dim=-1)

    Notes
    -----
    * ``empirical_normalization`` in the PPO config **must be set to False** when
      using equivariant modules.  Element-wise mean-subtraction of vector channels
      breaks SO(3) equivariance; the extractor should handle normalisation by
      design (e.g. using relative vectors whose expected mean is zero).
    """

    irreps_out: o3.Irreps

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim

    @abstractmethod
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Map flat obs ``(..., obs_dim)`` to an irrep-structured tensor ``(..., irreps_out.dim)``."""
        ...
