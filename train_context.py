import pickle
import numpy as np
import torch
import argparse
import gym
import os
import time
start_time = time.time()
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from context.model import RNNContextEncoder, RewardDecoder, StateDecoder, GeneralEncoder
from context.dataset import ContextDataset
from configs import args_point_robot ,args_half_cheetah_vel, args_half_cheetah_dir, args_ant_dir, args_hopper, args_walker, args_reach
from src.envs import PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv, AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='ant_dir')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--context_horizon', type=int, default=4)
args, rest_args = parser.parse_known_args()
env_type = args.env_type
context_horizon = args.context_horizon
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

if env_type == 'point_robot':
    args = args_point_robot.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'cheetah_vel':
    args = args_half_cheetah_vel.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'cheetah_dir':
    args = args_half_cheetah_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'ant_dir':
    args = args_ant_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'hopper':
    args = args_hopper.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'walker':
    args = args_walker.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'reach':
    args = args_reach.get_args(rest_args)
    args.context_horizon = context_horizon
else:
    raise NotImplementedError

torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=4, suppress=True)

# Environment
# make env, multi-task setting
if env_type == 'point_robot':
    env = PointEnv(max_episode_steps=args.max_episode_steps, num_tasks=args.num_tasks)
    env.seed(args.seed)
    env.load_all_tasks(np.load('./datasets/PointRobot-v0/task_goals.npy'))
elif env_type =='cheetah_vel':
   
    with open('./datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['velocity'] for item in velocities])
    # tasks = [{'velocity': velocity} for velocity in velocities]
    env=HalfCheetahVelEnv(tasks=velocities)
elif env_type=='ant_dir':
    with open('./datasets/AntDir-v0/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['goal'] for item in velocities])


    # tasks = [{'goal': velocity} for velocity in velocities]
    env=AntDirEnv(tasks=velocities)
elif env_type=='walker':
    with open('./datasets/WalkerRandParams-v0/task_goals.pkl', 'rb') as file:
        goals = pickle.load(file)
    
    env = WalkerRandParamsWrappedEnv(tasks=goals)
elif env_type == 'reach':
    with open('./datasets/Reach/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = ReachEnv(tasks=tasks)
elif env_type == 'cheetah_dir':
    with open('./datasets/HalfCheetahDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahDirEnv(tasks=tasks)
elif env_type == 'hopper':
    with open('./datasets/HopperRandParams-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HopperRandParamsEnv(tasks=tasks)


    

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
env.seed(args.seed)
if ((env_type!='walker')and(env_type!='hopper')):
    env.action_space.seed(args.seed)


########################################################################
### preparing the training and testing datasets
data_path = f'datasets/{args.env_name}/{args.data_quality}'
train_data, test_data = OrderedDict(), OrderedDict()

keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'masks']
if env_type=='cheetah_dir':
    if args.data_quality=='medium':
        keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'masks']
for key in keys:
    train_data[key] = []
    test_data[key] = []
for task_id in range(args.num_tasks):
    with open(f'{data_path}/dataset_task_{task_id}.pkl', "rb") as f:
        data = pickle.load(f)
    f.close()

    for key, values in data.items():
        if task_id < args.num_train_tasks:
            train_data[key].append(values)
        else:
            test_data[key].append(values)

for key, values in train_data.items():
    train_data[key] = np.concatenate(train_data[key], axis=0)
    test_data[key] = np.concatenate(test_data[key], axis=0)
if env_type=='cheetah_dir':
    if args.data_quality=='medium':
        train_data['observations']=train_data['states']
        train_data['next_observations']=train_data['next_states']
        train_data['terminals']=train_data['dones']
        test_data['observations'] = test_data['states']
        test_data['next_observations'] = test_data['next_states']
        test_data['terminals'] = test_data['dones']

########################################################################


train_dataset = ContextDataset(train_data, horizon=args.context_horizon, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=args.context_batch_size, shuffle=True)

test_dataset = ContextDataset(test_data, horizon=args.context_horizon, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=args.context_batch_size, shuffle=True)

# Tesnorboard
writer = SummaryWriter(f'runs/{args.env_name}/context/{args.data_quality}/horizon{args.context_horizon}')

# The models
# context_encoder = RNNContextEncoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
# reward_decoder = RewardDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
if ((env_type=='walker')or(env_type=='hopper')):
    context_encoder =RNNContextEncoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
    state_decoder = StateDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
else:
    context_encoder = RNNContextEncoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
    reward_decoder = RewardDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
if ((env_type=='walker')or(env_type=='hopper')):
    optimizer = torch.optim.Adam([*context_encoder.parameters(), *state_decoder.parameters()], lr=args.context_lr)
else:
    optimizer = torch.optim.Adam([*context_encoder.parameters(), *reward_decoder.parameters()], lr=args.context_lr)


save_model_path = f'saves/{args.env_name}/context/{args.data_quality}/{args.seed}/horizon{args.context_horizon}'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

global_step = 0
best_loss = float('inf')
# for epoch in range(args.context_train_epochs):
if ((env_type!='walker')and(env_type!='hopper')):
    for epoch in range(args.context_train_epochs):
        print(f'\n========== Epoch {epoch+1} ==========')

        # Model training
        
        context_encoder.train(); reward_decoder.train()
        for step, (transition, segment, next_segment) in tqdm(enumerate(train_dataloader)):
            state, action, reward, next_state, _, _ = transition 
            state_segment, action_segment, reward_segment = segment

            context = context_encoder(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1))
            reward_predict = reward_decoder(state, action, next_state, context)

            loss = F.mse_loss(reward_predict, reward)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*context_encoder.parameters(), *reward_decoder.parameters()], 1.0)
            optimizer.step()

            global_step += 1
            writer.add_scalar('loss/train', loss.item(), global_step)
            
        # model evaluation
        with torch.no_grad():
            context_encoder.eval(); reward_decoder.eval()
            transition, segment, next_segment = next(iter(test_dataloader))
            state, action, reward, next_state, _, _ = transition 
            state_segment, action_segment, reward_segment = segment

            context = context_encoder(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1))
            reward_predict = reward_decoder(state, action, next_state, context)

            loss = F.mse_loss(reward_predict, reward).detach().cpu().numpy()
            writer.add_scalar('loss/test', loss, epoch+1)           
            print(f'Model Evaluation, test loss: {loss}')

            if loss < best_loss:
                best_loss = loss 
                torch.save(
                    {'context_encoder': context_encoder.state_dict(), 'reward_decoder': reward_decoder.state_dict()}, 
                    f'{save_model_path}/context_models_best.pt'
                )
                print('Save the best model...')

            print(f'Predicted rewards: {reward_predict.detach().cpu().numpy()[:8].reshape(-1)}')
            print(f'   Real rewards  : {reward.detach().cpu().numpy()[:8].reshape(-1)}')

        if (epoch+1) % args.save_context_model_every ==0:
            torch.save(
                {'context_encoder': context_encoder.state_dict(), 'reward_decoder': reward_decoder.state_dict()}, 
                f'{save_model_path}/context_models_{epoch+1}.pt'
            )

        print(f'\nElapsed time: {round((time.time()-start_time)/60., 2)} minutes')
else:
    for epoch in range(args.context_train_epochs):
        print(f'\n========== Epoch {epoch+1} ==========')

        # Model training
        
        context_encoder.train(); state_decoder.train()
        for step, (transition, segment, next_segment) in tqdm(enumerate(train_dataloader)):
            state, action, reward, next_state, _, _ = transition 
            state_segment, action_segment, reward_segment = segment
            next_state_segment,_,_ = next_segment
            


            # context = context_encoder(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1),next_state_segment.transpose(0,1))
            context = context_encoder(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1))
            state_predict = state_decoder(state, action, reward, next_state,context)

            loss = F.mse_loss(state_predict,next_state)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*context_encoder.parameters(), *state_decoder.parameters()], 1.0)
            optimizer.step()

            global_step += 1
            writer.add_scalar('loss/train', loss.item(), global_step)
            
        # model evaluation
        with torch.no_grad():
            context_encoder.eval(); state_decoder.eval()
            transition, segment, next_segment = next(iter(test_dataloader))
            state, action, reward, next_state, _, _ = transition 
            # state = (state-means)/std
            # next_state = (next_state - means)/std

            state_segment, action_segment, reward_segment = segment
            next_state_segment,_,_ = next_segment

            context = context_encoder(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1))
            state_predict = state_decoder(state, action, reward, next_state,context)

            loss = F.mse_loss(state_predict, next_state).detach().cpu().numpy()
            writer.add_scalar('loss/test', loss, epoch+1)           
            print(f'Model Evaluation, test loss: {loss}')

            if loss < best_loss:
                best_loss = loss 
                torch.save(
                    {'context_encoder': context_encoder.state_dict(), 'state_decoder': state_decoder.state_dict()}, 
                    f'{save_model_path}/context_models_best.pt'
                )
                print('Save the best model...')

            print(f'Predicted state: {state_predict.detach().cpu().numpy()[:8].reshape(-1)}')
            print(f'   Real state  : {state.detach().cpu().numpy()[:8].reshape(-1)}')

        if (epoch+1) % args.save_context_model_every ==0:
            torch.save(
                {'context_encoder': context_encoder.state_dict(), 'state_decoder': state_decoder.state_dict()}, 
                f'{save_model_path}/context_models_{epoch+1}.pt'
            )

        print(f'\nElapsed time: {round((time.time()-start_time)/60., 2)} minutes')