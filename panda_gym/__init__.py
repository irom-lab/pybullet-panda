import gym

gym.envs.register(  # no time limit imposed
    id='GraspMultiView-v0',
    entry_point='panda_gym.grasp_mv_env:GraspMultiViewEnv',
)
