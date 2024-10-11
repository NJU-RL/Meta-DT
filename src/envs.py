import numpy as np
from typing import Optional, Tuple, List
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from gym.spaces import Box
from gym.wrappers import TimeLimit
from copy import deepcopy
import metaworld
import numpy as np
from typing import Optional, Tuple, List, Dict
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv as HopperRandParamsEnv_
from gym import spaces
from gym import Env
from gym.spaces import Box
from gym.wrappers import TimeLimit
from copy import deepcopy


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


class ML45Env(object):
    def __init__(self, include_goal: bool = False):
        self.n_tasks = 50
        self.tasks = list(HARD_MODE_ARGS_KWARGS['train'].keys()) + list(HARD_MODE_ARGS_KWARGS['test'].keys())

        self._max_episode_steps = 150

        self.include_goal = include_goal
        self._task_idx = None
        self._env = None
        self._envs = []

        _cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
        _args_kwargs = {**HARD_MODE_ARGS_KWARGS['train'], **HARD_MODE_ARGS_KWARGS['test']}
        for idx in range(self.n_tasks):
            task = self.tasks[idx]
            args_kwargs = _args_kwargs[task]
            if idx == 28 or idx == 29:
                args_kwargs['kwargs']['obs_type'] = 'plain'
                args_kwargs['kwargs']['random_init'] = False
            else:
                args_kwargs['kwargs']['obs_type'] = 'with_goal'
            args_kwargs['task'] = task
            env = _cls_dict[task](*args_kwargs['args'], **args_kwargs['kwargs'])
            self._envs.append(TimeLimit(env, max_episode_steps=self._max_episode_steps))
        
        self.set_task_idx(0)

    @property
    def observation_space(self):
        space = self._env.observation_space
        if self.include_goal:
            space = Box(low=space.low[0], high=space.high[0], shape=(space.shape[0] + len(self.tasks),))
        return space

    def reset(self):
        obs = self._env.reset()
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            obs = np.concatenate([obs, one_hot])
        return obs

    def step(self, action):
        o, r, d, i = self._env.step(action)
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            o = np.concatenate([o, one_hot])
        return o, r, d, i

    def set_task_idx(self, idx):
        self._task_idx = idx
        self._env = self._envs[idx]

    def __getattribute__(self, name):
        '''
        If we try to access attributes that only exist in the env, return the
        env implementation.
        '''
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            e_ = e
            try:
                return object.__getattribute__(self._env, name)
            except AttributeError as env_exception:
                pass
            except Exception as env_exception:
                e_ = env_exception
        raise e_


class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, tasks: List[dict] = None, n_tasks: int = None):
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        super().__init__(tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200
        
    
    def sample_tasks(self, num_tasks):
        directions = [(-1) ** n for n in range(num_tasks)]
        tasks = [{'direction': direction} for direction in directions]
        return tasks


    def _get_obs(self):
        obs = super()._get_obs()
        return obs


    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['direction']
        self.reset()


    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])
        

class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        super().__init__(tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            if self.one_hot_goal:
                goal = np.zeros((self.n_tasks,))
                goal[self.tasks.index(self._task)] = 1
            else:
                goal = np.array([self._goal_vel])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()

        return obs
        
    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])
    def print_task(self):
        print(f'Task information: Goal vel {self._goal}')
    def load_all_tasks(self, goals):
        # assert self.num_tasks == len(goals)
        self.goals = np.array([g for g in goals])
        self.reset_task(0)
    

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200
    
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
        

######################################################
######################################################
# <BEGIN DEPRECATED> #################################
######################################################
######################################################
class AntGoalEnv(AntGoalEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False,
                 reward_offset: float = 0.0, can_die: bool = False):
        self.include_goal = include_goal
        self.reward_offset = reward_offset
        self.can_die = can_die
        super().__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        self.task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 2
    
    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, self._goal])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']        
        self.reset()
        
class HumanoidDirEnv(HumanoidDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(HumanoidDirEnv, self).__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 1
        
    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, np.array([np.cos(self._goal), np.sin(self._goal)])])
        else:
            obs = super()._get_obs()
        return obs
    
    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done == True:
            rew = rew - 5.0
            done = False
        return (obs, rew, done, info)
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
######################################################
######################################################
# </END DEPRECATED> ##################################
######################################################
######################################################

class WalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv_):
    def __init__(self, tasks: List[dict] = None, n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        self.n_tasks = len(tasks) if tasks is not None else n_tasks
        
        super(WalkerRandParamsWrappedEnv, self).__init__(tasks, n_tasks)

        self.set_task_idx(0)
        self._max_episode_steps = 200
        
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
            one_hot = np.zeros(self.n_tasks, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
        
    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
class ReachEnv:
    def __init__(self, n_tasks: int = 20, tasks: List = None) -> None:
        self.env_list = []
        self._max_episode_step = 500
        
        reach_env = metaworld.ML1('reach-v2', 3407)
        if tasks is None:
            self.n_tasks = n_tasks
            self.tasks = []
            for i in range(n_tasks):
                env = reach_env.train_classes['reach-v2']()
                env.max_path_length = self._max_episode_step
                task = reach_env.train_tasks[i]
                env.set_task(task)
                self.env_list.append(env)
                self.tasks.append(task)
        else:
            self.tasks = tasks
            self.n_tasks = len(tasks)
            for task in tasks:
                env = reach_env.train_classes['reach-v2']()
                env.max_path_length = self._max_episode_step
                env.set_task(task)
                self.env_list.append(env)
        
        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(39,))
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
        
        
    def seed(self, seed: int):
        for env in self.env_list:
            env.seed(seed)


    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal_idx = idx
        # self._goal = self.tasks[idx]
        self._env = self.env_list[idx]
        self.reset()


    def reset(self):
        self.step_count = 0
        return self.reset_model()


    def reset_model(self):
        self._state = self._env.reset()
        return self._state


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state, reward, done, info = self._env.step(action)
        return next_state, np.array(reward), done, info
    

    def get_all_task_idx(self):
        return range(self.n_tasks)
class HopperRandParamsEnv(HopperRandParamsEnv_):
    def __init__(self, n_tasks=2, tasks: Dict = None, randomize_tasks=True, max_episode_steps=200):
        super(HopperRandParamsEnv, self).__init__()
        self.randomize_tasks = randomize_tasks
        if tasks is None:
            assert n_tasks is not None
            self.n_tasks = n_tasks
            self.tasks = self.sample_tasks(n_tasks)
        else:
            self.tasks = tasks
            self.n_tasks = len(tasks)
        self.reset_task(0)
        self._max_episode_steps = max_episode_steps
    
    
    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        np.random.seed(1234)
        param_sets = []
        if self.randomize_tasks:
            for i in range(n_tasks):
            
                new_params = {}
                # body mass -> one multiplier for all body parts
                if 'body_mass' in self.rand_params:
                    body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                    new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

                # body_inertia
                if 'body_inertia' in self.rand_params:
                    body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                    new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

                # damping -> different multiplier for different dofs/joints
                if 'dof_damping' in self.rand_params:
                    dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                    new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            	# friction at the body components
                if 'geom_friction' in self.rand_params:
                    dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                    new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

                param_sets.append(new_params)
        else:
            raise NotImplementedError

        return param_sets
    
    
    def get_all_task_idx(self):
        return range(len(self.tasks))


    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
