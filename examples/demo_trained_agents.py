import argparse
import os

from tactile_gym.sb3_helpers.eval_agent import final_evaluation

parser = argparse.ArgumentParser()

parser.add_argument(
    "-algo",
    type=str,
    default='ppo',
    help='Options: {ppo, sac}'
)

parser.add_argument(
    "-env",
    type=str,
    default='surface_follow-v0',
    help="""Options: {
            edge_follow-v0, surface_follow-v0, surface_follow-v1,
            object_roll-v0, object_push-v0, object_balance-v0}"""
)

parser.add_argument(
    "-obs",
    type=str,
    default='tactile',
    help='Options: {oracle, tactile, visual, visuotactile}'
)

# set args
args = parser.parse_args()
algo_name = args.algo
env_name = args.env
obs_type = args.obs

# evaluate params
n_eval_episodes = 10
seed = None
deterministic = True
show_gui = False
show_tactile = True
show_vision = True
render = True

# combine args
saved_model_dir = os.path.join(
    os.path.dirname(__file__), "../tactile_gym/sb3_helpers/saved_models",
    env_name,
    algo_name,
    obs_type
)

# run the evaluation
final_evaluation(
    saved_model_dir,
    n_eval_episodes,
    seed=seed,
    deterministic=deterministic,
    show_gui=show_gui,
    show_tactile=show_tactile,
    show_vision=show_vision,
    render=render,
    save_vid=False,
    take_snapshot=False
)
