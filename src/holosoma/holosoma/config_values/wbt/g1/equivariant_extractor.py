"""SO(3)-equivariant feature extractors for the G1-29DOF Whole Body Tracking task.

Why equivariance helps for WBT
-------------------------------
The actor receives observations that are already expressed **in the robot's base frame**
(torso_link).  This means the relevant symmetry is:

    If you rotate the entire scenario (robot + reference motion) about the
    VERTICAL axis (yaw), the required joint-position corrections are unchanged.

A standard MLP must *learn* this invariance from data.  An equivariant network
*enforces* it by construction, leading to:

* Fewer samples needed to generalise across initial headings / yaw-noisy resets.
* No spurious sensitivity to absolute world orientation.
* Cleaner geometric reasoning in the hidden layers.

Observation layout (actor_obs, dim = 154)
------------------------------------------
As defined in ``config_values/wbt/g1/observation.py`` with ``history_length=1``:

======  =========  =====  ===================================================
Slice   Term       Dim    Geometric role
======  =========  =====  ===================================================
0:58    motion_command (= [ref_joint_pos(29), ref_joint_vel(29)])
                    58    scalar joint targets from motion clip  → 0e
58:64   motion_ref_ori_b
                     6    6-D orientation error of the torso
                          (first 2 columns of the 3×3 relative rotation mat,
                           stored interleaved: col0_x, col1_x, col0_y, …)
                          → 2 × 1o polar vectors
64:67   base_ang_vel 3    angular velocity in base frame        → 1 × 1o
67:96   dof_pos     29    joint positions minus default         → 0e
96:125  dof_vel     29    joint velocities                      → 0e
125:154 actions     29    last policy actions                   → 0e
======  =========  =====  ===================================================

Equivariant irreps decomposition
----------------------------------
Vectors (1o) come first, then scalars (0e), as required by
:class:`~holosoma.agents.modules.equivariant.EquivariantFlatExtractor`:

    irreps_out = "3x1o + 145x0e"
    dim = 3*3 + 145 = 9 + 145 = 154  ✓

The scalar augmentation step inside
:class:`~holosoma.agents.modules.equivariant.EquivariantPPOActor`
automatically appends the 3 vector norms and the 3 pairwise dot products,
giving the Gate nonlinearity the geometric context it needs.

Important: set ``empirical_normalization: false`` in the PPO config.
Mean-subtracting the flat observation would corrupt the vector channels.
"""

from __future__ import annotations

import torch
from e3nn import o3
from holosoma.agents.modules.equivariant.extractor import EquivariantFlatExtractor


class G1WBTActorExtractor(EquivariantFlatExtractor):
    """Equivariant extractor for the G1-29DOF WBT actor observation (dim=154).

    Parses the flat observation into an irrep-structured tensor of shape
    ``(..., irreps_out.dim)`` = ``(..., 154)`` with the convention
    *vectors first, scalars after*:

    Output layout::

        [col0(3) | col1(3) | ang_vel(3) | motion_cmd(58) | dof_pos(29) | dof_vel(29) | actions(29)]
          1o       1o         1o           0e                0e            0e             0e

    The two orientation-error columns ``col0`` and ``col1`` are the first and
    second columns of the 3×3 relative rotation matrix (torso_target vs.
    torso_current), expressed in the base frame.  Together they form a 6-D
    rotation representation that transforms equivariantly under SO(3).

    Under a rotation R of the base frame:
    * col0 → R·col0,  col1 → R·col1,  ang_vel → R·ang_vel
    * all scalars are unchanged
    * the network's output (29 joint corrections) is also unchanged  ✓
    """

    irreps_out: o3.Irreps = o3.Irreps("3x1o + 145x0e")

    # Expected input dimension
    EXPECTED_OBS_DIM: int = 154

    # Flat index slices (must match the term ordering in actor_obs_shared)
    _IDX_MOTION_CMD = slice(0, 58)  # ref_joint_pos(29) + ref_joint_vel(29)
    _IDX_REF_ORI_6D = slice(58, 64)  # motion_ref_ori_b: interleaved 2-col rotation
    _IDX_ANG_VEL = slice(64, 67)  # base_ang_vel
    _IDX_DOF_POS = slice(67, 96)  # dof_pos
    _IDX_DOF_VEL = slice(96, 125)  # dof_vel
    _IDX_ACTIONS = slice(125, 154)  # actions

    def __init__(self, obs_dim: int = EXPECTED_OBS_DIM) -> None:
        if obs_dim != self.EXPECTED_OBS_DIM:
            raise ValueError(
                f"{self.__class__.__name__} expects obs_dim={self.EXPECTED_OBS_DIM}, got {obs_dim}. "
                "Check that the actor observation config matches the expected layout."
            )
        super().__init__(obs_dim)

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Map flat actor obs ``(..., 154)`` → irrep tensor ``(..., 154)``.

        The dimension is the same but the layout changes: vectors are moved
        to the front so that ``LinearBlock`` sees them as proper irreps.
        """
        # ── orientation error of the torso (motion_ref_ori_b) ────────────
        # The observation stores the first 2 columns of the 3×3 relative
        # rotation matrix in interleaved C-order:
        #   [col0_x, col1_x, col0_y, col1_y, col0_z, col1_z]
        # Reshape to (..., 3, 2) and split along the last axis.
        ref_ori_6d = flat_obs[..., self._IDX_REF_ORI_6D]  # (..., 6)
        mat_2col = ref_ori_6d.reshape(flat_obs.shape[:-1] + (3, 2))  # (..., 3, 2)
        col0 = mat_2col[..., 0]  # (..., 3) → 1o
        col1 = mat_2col[..., 1]  # (..., 3) → 1o

        # ── angular velocity (already a 3-D vector in base frame) ────────
        ang_vel = flat_obs[..., self._IDX_ANG_VEL]  # (..., 3) → 1o

        # ── scalar channels ───────────────────────────────────────────────
        motion_cmd = flat_obs[..., self._IDX_MOTION_CMD]  # (..., 58) → 0e
        dof_pos = flat_obs[..., self._IDX_DOF_POS]  # (..., 29) → 0e
        dof_vel = flat_obs[..., self._IDX_DOF_VEL]  # (..., 29) → 0e
        actions = flat_obs[..., self._IDX_ACTIONS]  # (..., 29) → 0e

        # Layout: vectors (1o) first, scalars (0e) after — required by the
        # equivariant architecture so that irreps_out is correctly interpreted.
        return torch.cat([col0, col1, ang_vel, motion_cmd, dof_pos, dof_vel, actions], dim=-1)
