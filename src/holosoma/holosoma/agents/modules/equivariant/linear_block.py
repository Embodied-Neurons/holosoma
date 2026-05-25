"""Gate-based equivariant linear layer using e3nn."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate


class LinearBlock(nn.Module):
    """Equivariant linear layer followed by a Gate nonlinearity.

    Maps ``irreps_in -> n_scalars x 0e + n_vectors x 1o`` in two steps:

    1. ``o3.Linear`` maps the input irreps to the pre-gate layout required by
       ``Gate``: ``[n_scalars x 0e | n_vectors x 0e (gate) | n_vectors x 1o]``
    2. ``Gate`` applies ``tanh`` to the scalars and ``sigmoid``-gated multiplication
       to the vectors (element-wise gate scalar × vector).

    The gate scalars are *consumed* by the Gate so the output layout is simply
    ``n_scalars x 0e + n_vectors x 1o``.

    This is the building block used in the equivariant SAC actor example and
    follows Weiler & Cesa (2019) / e3nn conventions.
    """

    def __init__(self, irreps_in: o3.Irreps, n_scalars: int, n_vectors: int) -> None:
        super().__init__()

        irreps_scalars = o3.Irreps(f"{n_scalars}x0e")  # pass-through scalars activated by tanh
        irreps_gates = o3.Irreps(f"{n_vectors}x0e")  # one gate scalar per output vector
        irreps_gated = o3.Irreps(f"{n_vectors}x1o")  # actual equivariant vector outputs

        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.tanh],
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid],
            irreps_gated=irreps_gated,
        )

        # o3.Linear maps input irreps -> gate input layout (scalars + gate_scalars + gated)
        self.linear = o3.Linear(irreps_in, self.gate.irreps_in)

        # Expose final irreps for stacking blocks
        self.irreps_out: o3.Irreps = self.gate.irreps_out  # "n_scalars x 0e + n_vectors x 1o"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.linear(x))
