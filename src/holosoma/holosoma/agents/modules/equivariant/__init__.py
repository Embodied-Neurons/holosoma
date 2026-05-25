"""Equivariant PPO building blocks.

Usage
-----
To use equivariant modules in PPO, set ``type: "Equivariant"`` in the actor
and critic ``ModuleConfig`` and fill in the equivariant fields of
``LayerConfig``::

    from holosoma.agents.modules.equivariant import EquivariantFlatExtractor
    from e3nn import o3

    class MyExtractor(EquivariantFlatExtractor):
        irreps_out = o3.Irreps("2x1o + 4x0e")

        def __init__(self, obs_dim: int) -> None:
            super().__init__(obs_dim)

        def forward(self, flat_obs):
            ...  # parse flat_obs -> irrep-structured tensor
"""

from .extractor import EquivariantFlatExtractor
from .linear_block import LinearBlock
from .ppo_modules import EquivariantPPOActor, InvariantPPOCritic

__all__ = [
    "EquivariantFlatExtractor",
    "LinearBlock",
    "EquivariantPPOActor",
    "InvariantPPOCritic",
]
