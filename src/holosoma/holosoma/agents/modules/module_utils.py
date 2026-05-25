from __future__ import annotations

from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCritic, PPOCriticEncoder
from holosoma.utils.helpers import get_class


def setup_ppo_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)
    if module_type == "Equivariant":
        return _setup_equivariant_actor(
            obs_dim_dict=obs_dim_dict,
            module_config=module_config,
            init_noise_std=init_noise_std,
            device=device,
        )

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict,
    module_config,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    if module_type == "Equivariant":
        return _setup_equivariant_critic(
            obs_dim_dict=obs_dim_dict,
            module_config=module_config,
            device=device,
        )
    raise ValueError(f"Invalid critic type: {module_type}")


# ---------------------------------------------------------------------------
# Equivariant module builders
# ---------------------------------------------------------------------------


def _setup_equivariant_actor(obs_dim_dict, module_config, init_noise_std, device):
    """Build an :class:`~holosoma.agents.modules.equivariant.EquivariantPPOActor`.

    Called when ``ModuleConfig.type == "Equivariant"``.
    Reads extractor class, action irreps, and hidden sizes from
    ``module_config.layer_config``.
    """
    from holosoma.agents.modules.equivariant.ppo_modules import EquivariantPPOActor

    layer_cfg = module_config.layer_config

    if not layer_cfg.equivariant_extractor_class:
        raise ValueError(
            "layer_config.equivariant_extractor_class must be set when ModuleConfig.type == 'Equivariant'."
        )

    extractor_cls = get_class(layer_cfg.equivariant_extractor_class)
    obs_dim = sum(obs_dim_dict[k] for k in module_config.input_dim)
    extractor = extractor_cls(obs_dim=obs_dim)

    return EquivariantPPOActor(
        extractor=extractor,
        action_irreps_str=layer_cfg.equivariant_action_irreps,
        n_scalars=layer_cfg.equivariant_n_scalars,
        n_vectors=layer_cfg.equivariant_n_vectors,
        init_noise_std=init_noise_std,
    ).to(device)


def _setup_equivariant_critic(obs_dim_dict, module_config, device):
    """Build an :class:`~holosoma.agents.modules.equivariant.InvariantPPOCritic`.

    Called when ``ModuleConfig.type == "Equivariant"``.
    The critic uses rotation-invariant features (norms + dot products of
    all vector channels) fed through a plain MLP.
    """
    from holosoma.agents.modules.equivariant.ppo_modules import InvariantPPOCritic

    layer_cfg = module_config.layer_config

    if not layer_cfg.equivariant_extractor_class:
        raise ValueError(
            "layer_config.equivariant_extractor_class must be set when ModuleConfig.type == 'Equivariant'."
        )

    extractor_cls = get_class(layer_cfg.equivariant_extractor_class)
    obs_dim = sum(obs_dim_dict[k] for k in module_config.input_dim)
    extractor = extractor_cls(obs_dim=obs_dim)

    return InvariantPPOCritic(
        extractor=extractor,
        hidden_dims=tuple(layer_cfg.equivariant_critic_hidden_dims),
    ).to(device)
