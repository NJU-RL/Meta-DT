from gym.envs.registration import register


# point robot with fixed initial state
register(
    'PointRobot-v0',
    entry_point='environments.point_robot:PointEnv',
    max_episode_steps=20,
)





