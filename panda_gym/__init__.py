import gym


gym.envs.register(  # no time limit imposed
    id='GraspMultiView-v0',
    entry_point='panda_gym.grasp_mv_env:GraspMultiViewEnv',
)

gym.envs.register(  # no time limit imposed
    id='GraspMultiViewRandom-v0',
    entry_point='panda_gym.grasp_mv_random_env:GraspMultiViewRandomEnv',
)

gym.envs.register(  # no time limit imposed
    id='Push-v0',
    entry_point='panda_gym.push_env:PushEnv',
)

gym.envs.register(  # no time limit imposed
    id='PushTool-v0',
    entry_point='panda_gym.push_tool_env:PushToolEnv',
)

gym.envs.register(  # no time limit imposed
    id='Lift-v0',
    entry_point='panda_gym.lift_env:LiftEnv',
)

gym.envs.register(  # no time limit imposed
    id='Hammer-v0',
    entry_point='panda_gym.hammer_env:HammerEnv',
)

gym.envs.register(  # no time limit imposed
    id='Sweep-v0',
    entry_point='panda_gym.sweep_env:SweepEnv',
)


