import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor

env_args = {
    "env_params": {
        "max_steps": 200,
        "show_gui": False,

        "observation_mode": "oracle",
        # "observation_mode": "tactile",
        # "observation_mode": "visual",
        # "observation_mode": "visuotactile",
        # "observation_mode": "tactile_and_feature",
        # "observation_mode": "visual_and_feature",
        # "observation_mode": "visuotactile_and_feature",
    },
    "robot_arm_params": {
        "type": "ur5",
        # "type": "franka_panda",
        # "type": "kuka_iiwa",
        # "type": "cr3",
        # "type": "mg400",

        # "control_mode": "tcp_position_control",
        "control_mode": "tcp_velocity_control",
        "control_dofs": ["x", "y", "z", "Rx", "Ry", "z"],

        # the type of control used
        # "control_mode": "joint_position_control",
        # "control_mode": "joint_velocity_control",
        # "control_dofs": ["J1", "J2", "J3", "J4", "J5", "J6"],
        # "control_dofs": ["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
    },
    "tactile_sensor_params": {
        "type": "standard_tactip",
        # "type": "standard_digit",
        # "type": "standard_digitac",

        "image_size": [128, 128],
        "turn_off_border": False,
        "show_tactile": False,
    },
    "visual_sensor_params": {
        "image_size": [128, 128],
        "show_vision": False
    }
}

rl_params_ppo = {
    "algo_name": "ppo",
    "env_id": None,
    "policy": "MultiInputPolicy",
    "seed": int(1),
    "n_stack": 1,
    "total_timesteps": int(1e6),
    "n_eval_episodes": 10,
    "n_envs": 10,
    "eval_freq": 2e3,
    "norm_obs": False,
    "norm_reward": False,
}

ppo_params = {
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            "cnn_base": NatureCNN,
            "cnn_output_dim": 256,
            "mlp_extractor_net_arch": [64, 64],
        },
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        "activation_fn": nn.Tanh,
    },
    "learning_rate": 3e-4,
    "n_steps": int(2048),
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": 0.1,
}


rl_params_sac = {
    "algo_name": "sac",
    "env_id": None,
    "policy": "MultiInputPolicy",
    "seed": int(1),
    "n_stack": 1,
    "total_timesteps": int(1e6),
    "n_eval_episodes": 10,
    "n_envs": 1,
    "eval_freq": 1e4,
    "norm_obs": False,
    "norm_reward": False,
}

sac_params = {
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            "cnn_base": NatureCNN,
            "cnn_output_dim": 256,
            "mlp_extractor_net_arch": [64, 64],
        },
        "net_arch": dict(pi=[256, 256], qf=[256, 256]),
        "activation_fn": nn.Tanh,
    },
    "learning_rate": 7.5e-5,
    "buffer_size": int(1e6),
    "learning_starts": 1e5,
    "batch_size": 2048,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "optimize_memory_usage": False,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
}
