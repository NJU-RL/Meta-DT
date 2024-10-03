import numpy as np
from environments.mujoco.rand_param_envs.base import RandomEnv
from environments.mujoco.rand_param_envs.gym import utils
import sys

class HopperRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, n_tasks=50, max_episode_steps=200, log_scale_limit=3.0, **kwargs):
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = -1  # the thing below takes one step

        RandomEnv.__init__(self, log_scale_limit, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)
        #print(len(self.tasks))
        #print(self.tasks[0])
        #for k in self.tasks[0]:
        #    print(k, self.tasks[0][k].shape)
        #self.max_episode_steps = max_episode_steps
        self.curr_params = self.tasks[0]
        #print(self.curr_params)
        self._goal = np.concatenate([self.curr_params[k].reshape(-1) for k in self.curr_params])


    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 0.0#1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #            (height > .7) and (abs(ang) < .2))
        done = False
        ob = self._get_obs()
        self._elapsed_steps += 1
        info = {'task': self.get_task()}
        if self._elapsed_steps == self._max_episode_steps:
            done = True
            info['bad_transition'] = True
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat, #[1:],
            self.model.data.qvel.flat
            #np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def _reset(self):
        ob = super()._reset()
        self._elapsed_steps = 0
        return ob

    def reset_task(self, idx=None):
        if idx is not None:
            self.curr_params = self.tasks[idx]
            self._goal = np.concatenate([self.curr_params[k].reshape(-1) for k in self.curr_params])
            self.set_task(self.curr_params)
        self.reset()

    def set_all_goals(self, goals):
        assert self.num_tasks == len(goals)
        for i in range(len(goals)):
            self.tasks[i] = {'body_mass': goals[i][0:5].reshape(5,1),
                'body_inertia': goals[i][5:20].reshape(5,3),
                'dof_damping': goals[i][20:26].reshape(6,1),
                'geom_friction': goals[i][26:].reshape(5,3)}
        self.reset_task(0)

    def set_goal(self, goal):
        self.tasks[0] = {'body_mass': goal[0:5].reshape(5,1),
                'body_inertia': goal[5:20].reshape(5,3),
                'dof_damping': goal[20:26].reshape(6,1),
                'geom_friction': goal[26:].reshape(5,3)}
        self.reset_task(0)
        self._goal = goal

    def get_all_task_idx(self):
        return range(len(self.tasks))
    def print_task(self):
        print(f'Task information: Goal  {self._goal}')

class HopperRandParamsOracleEnv(HopperRandParamsEnv):
    def _get_obs(self):
        if hasattr(self, 'cur_params'):
            task = self.get_task()
            task = np.concatenate([task[k].reshape(-1) for k in task.keys()])[:, np.newaxis]
        else:
            task = np.zeros((self.rand_param_dim, 1))
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10),
            task.flat,
        ])
