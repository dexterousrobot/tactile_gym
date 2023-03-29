import sys


def import_parameters(env_name, algo_name):

    if env_name == "example_arm-v0":
        from tactile_gym.sb3_helpers.params.example_arm_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    elif env_name == "edge_follow-v0":
        from tactile_gym.sb3_helpers.params.edge_follow_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    elif env_name == "surface_follow-v0":
        from tactile_gym.sb3_helpers.params.surface_follow_auto_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    elif env_name == "surface_follow-v1":
        from tactile_gym.sb3_helpers.params.surface_follow_goal_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )
    elif env_name == "object_roll-v0":
        from tactile_gym.sb3_helpers.params.object_roll_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    elif env_name == "object_push-v0":
        from tactile_gym.sb3_helpers.params.object_push_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    elif env_name == "object_balance-v0":
        from tactile_gym.sb3_helpers.params.object_balance_params import (
            env_args,
            rl_params_ppo,
            ppo_params,
            rl_params_sac,
            sac_params,
        )

    else:
        sys.exit("Incorrect environment specified: {}.".format(env_name))

    if not algo_name:
        return env_args
    elif algo_name == "ppo":
        return env_args, rl_params_ppo, ppo_params
    elif algo_name == "sac":
        return env_args, rl_params_sac, sac_params
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))


if __name__ == "__main__":
    env_args, rl_params, algo_params = import_parameters("edge_follow-v0", "ppo")
    env_args, rl_params, algo_params = import_parameters("surface_follow-v0", "ppo")
    env_args, rl_params, algo_params = import_parameters("surface_follow-v1", "ppo")
    env_args, rl_params, algo_params = import_parameters("object_roll-v0", "ppo")
    env_args, rl_params, algo_params = import_parameters("object_push-v0", "ppo")
    env_args, rl_params, algo_params = import_parameters("object_balance-v0", "ppo")
