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
