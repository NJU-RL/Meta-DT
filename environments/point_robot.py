import numpy as np
import torch
from gym import spaces
from gym import Env
import json


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
            self,
            max_episode_steps=20,
            num_tasks=50
        ):

        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.num_tasks = num_tasks
        self.goals = np.array([[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(num_tasks)])

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self.reset()

    def print_task(self):
        print(f'Task information: Goal position {self._goal}')

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def load_all_tasks(self, goals):
        assert self.num_tasks == len(goals)
        self.goals = np.array([g for g in goals])
        self.reset_task(0)

    def reset_model(self):
        self._state = np.zeros(2)
        return self._get_obs()

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]

        reward = - (x ** 2 + y ** 2) ** 0.5

        # done = (abs(x) < 0.01) and (abs(y) < 0.01)
        done = False

        ob = self._get_obs()
        return ob, reward, done, dict()

    def reward(self, state, action=None):
        return - ((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2) ** 0.5

    def render(self):
        print('current state:', self._state)





