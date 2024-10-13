import argparse
import gym
import json
import pickle
import torch
import numpy as np

from collections import OrderedDict
from pathlib import Path

from configs import args_point_robot, args_half_cheetah_vel, args_half_cheetah_dir, args_ant_dir, args_hopper, args_walker, args_reach
from data_collection.replay_memory import ReplayMemory
from data_collection.sac import SAC
from src.envs import PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv, AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv


def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    if local_args.env_type not in ['walker', 'hopper']:
        env.action_space.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--env_type', type=str, default='point_robot', help='environment')
parser.add_argument('--data_type', type=str, default='medium')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--task_id_start', type=int, default=0)
parser.add_argument('--task_id_end', type=int, default=5)
parser.add_argument('--suffix', type=int, default=2000, help='model checkpoint suffix')
parser.add_argument('--capacity', type=int, default=2000, help='total timesteps')
local_args, rest_args = parser.parse_known_args()

# download config
if local_args.env_type == 'point_robot':
    args = vars(args_point_robot.get_args(rest_args))
    tasks = np.load('./datasets/PointRobot-v0/task_goals.npy')
    env = PointEnv(max_episode_steps=args['max_episode_steps'], num_tasks=args['num_tasks'])
    env.load_all_tasks(tasks)
elif local_args.env_type == 'cheetah_vel':
    args = vars(args_half_cheetah_vel.get_args(rest_args))
    with open('./datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahVelEnv(tasks=tasks)
elif local_args.env_type == 'cheetah_dir':
    args = vars(args_half_cheetah_vel.get_args(rest_args))
    with open('./datasets/HalfCheetahDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahDirEnv(tasks=tasks)
elif local_args.env_type =='ant_dir':
    args = vars(args_ant_dir.get_args(rest_args))
    with open('./datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = AntDirEnv(tasks=tasks)
elif local_args.env_type == 'walker':
    args = vars(args_walker.get_args(rest_args))
    with open('./datasets/WalkerRandParams-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = WalkerRandParamsWrappedEnv(tasks=tasks)
elif local_args.env_type == 'hopper':
    args = vars(args_hopper.get_args(rest_args))
    with open('./datasets/HopperRandParams-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HopperRandParamsEnv(tasks=tasks)
elif local_args.env_type == 'reach':
    args = vars(args_reach.get_args(rest_args))
    with open('./datasets/Reach-v2/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = ReachEnv(tasks=tasks)
else:
    raise NotImplementedError
args['device'] = torch.device(local_args.device) if torch.cuda.is_available() else torch.device('cpu')
args['save_path'] = Path(f"./datasets/{args['env_name']}/{local_args.data_type}")
args['save_path'].mkdir(parents=True, exist_ok=True)

# environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min, action_max = env.action_space.low, env.action_space.high
action_abs_min = min(np.abs(action_min).min(), np.abs(action_max).min())

# set seed
set_seed(local_args.seed, env)

# task information
task_info = OrderedDict()
return_scale_info = OrderedDict()

if local_args.data_type == 'medium':
    if local_args.env_type == 'point_robot':
        local_args.suffix = [480] * args['num_tasks']
    elif local_args.env_type == 'cheetah_vel':
        local_args.suffix = [60000] * args['num_tasks']
    elif local_args.env_type == 'cheetah_dir':
        local_args.suffix = [80000, 40000, 80000, 40000]
    elif local_args.env_type == 'ant_dir':
        local_args.suffix = [
            120000, 130000, 110000, 130000, 160000, 140000, 140000, 130000, 120000, 120000, 
            100000, 100000, 120000, 120000, 120000, 100000, 110000, 110000, 120000, 110000, 
            110000, 120000, 130000, 130000, 130000, 120000, 110000, 110000, 120000, 140000, 
            120000, 120000, 120000, 140000, 100000, 130000, 110000, 120000, 130000, 130000, 
            110000, 130000, 150000, 110000, 140000, 110000, 120000, 130000, 120000, 120000
        ]
    elif local_args.env_type == 'hopper':
        local_args.suffix = [
            60000,  70000,  70000,  70000,  40000,  50000,  60000,  80000, 110000,  80000, 
            180000, 150000,  80000,  60000,  60000, 130000,  60000,  50000,  70000, 150000, 
            200000, 100000, 130000, 150000,  70000, 100000,  50000, 100000, 100000, 150000, 
            60000,  70000, 100000,  50000,  70000,  40000,  60000,  60000,  80000,  60000, 
            110000,  50000,  70000,  80000,  70000, 100000, 100000,  70000, 250000,  70000
        ]
    elif local_args.env_type == 'walker':
        local_args.suffix = [
            220000, 220000, 200000, 240000, 240000, 140000, 160000, 200000, 240000, 200000, 
            140000, 260000, 120000, 240000, 200000, 160000, 160000, 160000, 200000, 140000, 
            300000, 240000, 200000, 240000, 200000, 180000, 180000, 200000, 240000, 200000,
            160000, 220000, 180000, 200000, 240000, 260000, 200000, 170000, 240000, 320000, 
            160000, 200000, 240000, 260000, 200000, 160000, 160000, 140000, 240000, 220000
        ]
    elif local_args.env_type == 'reach':
        local_args.suffix = [
            23000, 23000,  8000, 23000, 35000, 12000,  6000, 13000, 11000, 10000, 
            8000, 14000,  7000, 11000, 10000, 10000, 21000,  6000, 36000, 17000
        ]
else:
    local_args.suffix = [local_args.suffix] * args['num_tasks']
print(local_args.suffix)

for task_id in range(local_args.task_id_start, local_args.task_id_end):
    model_path = f"./datasets/{args['env_name']}/checkpoints/task_{task_id}/agent_{local_args.suffix[task_id]}.pt"
    agent = SAC(env, args['hidden_dim'], args['alpha'], args['lr'], args['gamma'], args['tau'], args['device'])
    agent.load(model_path)
    
    replaybuffer = ReplayMemory(local_args.capacity, local_args.seed)
    episode_returns = []
    
    env.reset_task(task_id)
    total_timestep = 0
    while total_timestep < local_args.capacity:
        episode_return = 0.
        state = env.reset()
        for step in range(args['max_episode_steps']):
            action = agent.select_action(state, False)
            action = np.clip(action, action_min, action_max)
            next_state, reward, done, _ = env.step(action)
            mask = True if (step == args['max_episode_steps'] - 1) else (not done)
            replaybuffer.push(state, action, reward, next_state, done, mask)
            
            state = next_state
            total_timestep += 1
            episode_return += reward
            
            if done:
                break
        
        episode_returns.append(episode_return)
    
    replaybuffer.save_buffer(args['save_path'] / f"dataset_task_{task_id}.pkl")
    
    # task information
    if local_args.env_type == 'point_robot':
        task_info[f"task {task_id}"] = {
            'goal': list(env.goals[task_id].tolist()),
            'return_scale': [min(episode_returns), max(episode_returns)]
        }
        with open(args['save_path'] / f'task_info_{local_args.task_id_start}.json', 'w') as fp:
            json.dump(task_info, fp, indent=4)
    elif local_args.env_type in ['walker', 'hopper']:
        task_info[f"task {task_id}"] = {key: item.tolist() for key, item in env.tasks[task_id].items()}
        task_info[f"task {task_id}"].update({'return_scale': [min(episode_returns), max(episode_returns)]})
        with open(args['save_path'] / f'task_info_{local_args.task_id_start}.json', 'w') as fp:
            json.dump(task_info, fp, indent=4)
        return_scale_info[f"task {task_id}"] = {'return_scale': [min(episode_returns), max(episode_returns)]}
        with open(args['save_path'] / f"return_scale_info_{local_args.task_id_start}.json", 'w') as fp:
            json.dump(return_scale_info, fp, indent=4)
    elif local_args.env_type == 'reach':
        task_info[f"task {task_id}"] = {
            'return_scale': [min(episode_returns), max(episode_returns)]
        }
        with open(args['save_path'] / f'task_info_{local_args.task_id_start}.json', 'w') as fp:
            json.dump(task_info, fp, indent=4)
    else:
        task_info[f"task {task_id}"] = {
            'goal': list(env.tasks[task_id].values()),
            'return_scale': [min(episode_returns), max(episode_returns)]
        }
        with open(args['save_path'] / f'task_info_{local_args.task_id_start}.json', 'w') as fp:
            json.dump(task_info, fp, indent=4)