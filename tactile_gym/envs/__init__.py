from gym.envs.registration import register

register(
    id="example_arm-v0",
    entry_point="tactile_gym.envs.example_envs.example_arm_env.example_arm_env:ExampleArmEnv",
)

register(
    id="edge_follow-v0",
    entry_point="tactile_gym.envs.exploration.edge_follow.edge_follow_env:EdgeFollowEnv",
)

register(
    id="surface_follow-v0",
    entry_point="tactile_gym.envs.exploration.surface_follow.surface_follow_auto.surface_follow_auto_env:SurfaceFollowAutoEnv",
)

register(
    id="surface_follow-v1",
    entry_point="tactile_gym.envs.exploration.surface_follow.surface_follow_goal.surface_follow_goal_env:SurfaceFollowGoalEnv",
)

register(
    id="object_roll-v0",
    entry_point="tactile_gym.envs.nonprehensile_manipulation.object_roll.object_roll_env:ObjectRollEnv",
)

register(
    id="object_push-v0",
    entry_point="tactile_gym.envs.nonprehensile_manipulation.object_push.object_push_env:ObjectPushEnv",
)

register(
    id="object_balance-v0",
    entry_point="tactile_gym.envs.nonprehensile_manipulation.object_balance.object_balance_env:ObjectBalanceEnv",
)
