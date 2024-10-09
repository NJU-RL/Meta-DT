import argparse
import gym
import numpy as np
import os
import torch
import json
from tqdm import tqdm 
import time 
start_time = time.time()
import d4rl
import d4rl.gym_mujoco
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import pickle 
import json 
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from configs import args_point_robot,args_half_cheetah_vel,args_ant_dir,args_walker_para,args_meta_world,args_half_cheetah_dir,args_hopper
import environments 
from meta_dt.trainer import MetaDT_Trainer
from meta_dt.model import MetaDecisionTransformer
from meta_dt.dataset import MetaDT_Dataset, append_context_to_data,append_error_to_trajectory
from meta_dt.evaluation import meta_evaluate_episode_rtg
from decision_transformer.dataset import convert_data_to_trajectories,discount_cumsum
from context.model import RNNContextEncoder ,RewardDecoder,GeneralEncoder,StateDecoder
from src.envs import  HalfCheetahVelEnv,AntDirEnv,WalkerRandParamsWrappedEnv,HalfCheetahDirEnv,ReachEnv,HopperRandParamsEnv
import random


parser = argparse.ArgumentParser()
parser.add_argument('--env_type', type=str, default='walker_para')
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
elif  env_type == 'ant_dir':
    args = args_ant_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'walker_para':
    args = args_walker_para.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'reach':
    args = args_meta_world.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'cheetah_dir':
    args = args_half_cheetah_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'hopper':
    args = args_hopper.get_args(rest_args)
    args.context_horizon = context_horizon
else:
    raise NotImplementedError
results_dir = f'runs/{args.env_name}/{args.zero_shot}/{args.data_quality}/{args.seed}/horizon{args.context_horizon}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=3, suppress=True)


# make env, multi-task setting
if env_type == 'point_robot':
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.load_all_tasks(np.load(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.npy'))
elif env_type =='cheetah_vel':
   
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['velocity'] for item in velocities])
    # tasks = [{'velocity': velocity} for velocity in velocities]
    env=HalfCheetahVelEnv(tasks=velocities)
elif env_type=='ant_dir':
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['goal'] for item in velocities])


    # tasks = [{'goal': velocity} for velocity in velocities]
    env=AntDirEnv(tasks=velocities)
elif env_type=='walker_para':
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as file:
        goals = pickle.load(file)
    
    env = WalkerRandParamsWrappedEnv(tasks=goals)
elif env_type == 'reach':
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = ReachEnv(tasks=tasks)
elif env_type == 'cheetah_dir':
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahDirEnv(tasks=tasks)
elif env_type == 'hopper':
    with open(f'./datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HopperRandParamsEnv(tasks=tasks)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# load the task information
with open(f'datasets/{args.env_name}/{args.data_quality}/task_info.json', 'r') as f:
    task_info = json.load(f)
f.close()
train_task_ids = np.arange(args.num_train_tasks)
eval_train_task_ids = np.arange(5)
if env_type=='cheetah_dir':
    
    eval_train_task_ids = np.arange(2)
test_task_ids = np.arange(args.num_train_tasks, args.num_tasks)
prompt_trajectories_list = []  

for ind in range(args.num_tasks):
    with open(f'datasets/{args.env_name}/{args.data_quality}/dataset_task_prompt{ind}.pkl', "rb") as f:
        prompt_trajectories =pickle.load(f) 
    
    f.close()
    prompt_trajectories_list.append(prompt_trajectories)
# num_samples = train_data['observations'].shape[0]
# print(f'Number of training samples: {num_samples}')

### load the pretrained context encoder 
if ((env_type=='walker_para')or(env_type=='hopper')):
    context_encoder = RNNContextEncoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
    dynamic_decoder = StateDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
else:
    context_encoder = RNNContextEncoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
    dynamic_decoder = RewardDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
load_path = f'./saves/{args.env_name}/context/{args.data_quality}/horizon{args.context_horizon}/context_models_best.pt'
context_encoder.load_state_dict(torch.load(load_path)['context_encoder'])
if ((env_type=='walker_para')or(env_type=='hopper')):
    dynamic_decoder.load_state_dict(torch.load(load_path)['state_decoder'])
else:
    dynamic_decoder.load_state_dict(torch.load(load_path)['reward_decoder'])
for name, param in context_encoder.named_parameters():
    param.requires_grad = False 
for name, param in dynamic_decoder.named_parameters():
    param.requires_grad = False 
print('Load context encoder from {}'.format(load_path))
world_model = [context_encoder, dynamic_decoder]
### compute the context (z) using the pretrained context encoder for each transition (s, a, r, s')
# train_data = append_context_to_data(train_data, context_encoder, horizon=args.context_horizon, device=device)

### transform the data into trajectories, align with the dataset prepared for DT 
# train_trajectories = convert_data_to_trajectories(train_data)
train_trajectories = []
for task_id in train_task_ids:
    train_data = OrderedDict()
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals','masks']
    if env_type=='cheetah_dir':
        if args.data_quality=='medium':
            keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'masks']
    for key in keys:
        train_data[key] = []
    # with open(f'{data_path}/{env_type}-{task_id}-expert.pkl', "rb") as f:
    with open(f'datasets/{args.env_name}/{args.data_quality}/dataset_task_{task_id}.pkl', "rb") as f:
        data = pickle.load(f)
        
    
    
    for key, values in data.items():
    # for key, values in data:
        train_data[key].append(values)
    for key, values in train_data.items():
    # for key, values in train_data:
        train_data[key] = np.concatenate(values, axis=0)
    if env_type=='cheetah_dir':
        if args.data_quality=='medium':
            train_data['observations']=train_data['states']
            train_data['next_observations']=train_data['next_states']
            train_data['terminals']=train_data['dones']
        # test_data['observations'] = test_data['states']
        # test_data['next_observations'] = test_data['next_states']
        # test_data['terminals'] = test_data['dones']
    train_data = append_context_to_data(train_data, context_encoder, horizon=args.context_horizon, device=device,args=args)
    train_trajectories_per = convert_data_to_trajectories(train_data,args)
    for trajectory in train_trajectories_per:
        train_trajectories.append(trajectory)


train_dataset = MetaDT_Dataset(
    train_trajectories, 
    args.dt_horizon, 
    args.max_episode_steps, 
    args.dt_return_scale, 
    device,
    prompt_trajectories_list=prompt_trajectories_list,
    args=args,
    world_model = world_model
)
state_mean, state_std = train_dataset.state_mean, train_dataset.state_std

### save the arguments for debugging
variant = vars(args)
variant.update(version=f"Decision Transformer")
variant.update(state_dim=state_dim)
variant.update(action_dim=action_dim)
variant.update(return_min=float(train_dataset.return_min))
variant.update(return_max=float(train_dataset.return_max))
variant.update(return_avg=float(train_dataset.return_avg))

with open(os.path.join(results_dir, 'variant.json'), 'w') as f:
    f.write(json.dumps(variant, indent=4))
f.close() 

# sample_weights = [len(traj['observations']) for traj in train_dataset.trajectories]
# sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.meta_dt_batch_size, shuffle=True)

model = MetaDecisionTransformer(
    state_dim=state_dim,
    act_dim=action_dim,
    max_length=args.dt_horizon,
    max_ep_len=args.max_episode_steps,
    context_dim=args.context_dim,
    hidden_size=args.dt_embed_dim,
    n_layer=args.dt_n_layer,
    n_head=args.dt_n_head,
    n_inner=4*args.dt_embed_dim,
    activation_function=args.dt_activation_function,
    n_positions=1024,
    resid_pdrop=args.dt_dropout,
    attn_pdrop=args.dt_dropout,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.dt_lr,
    weight_decay=args.dt_weight_decay,
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps+1)/args.meta_dt_warmup_steps, 1)
)

agent = MetaDT_Trainer(model, optimizer)

writer =  SummaryWriter(results_dir)

global_step = 0
max_len = args.prompt_length
max_ep_len = args.max_ep_len
scale = args.scale
trajectories_buffer = []
for ids in range(args.num_tasks):
    traj = []

    trajectories_buffer.append(traj)
# pbar = tqdm(total=args.max_step)

while global_step<= args.max_step:
    
    print(f'\n==========  {global_step} ==========')

    for step, batch in tqdm(enumerate(train_dataloader)):
        # states, contexts, actions, rewards, dones, rtg, timesteps, masks = batch
        states, contexts, actions, rewards, dones, rtg, timesteps, masks,prompt_states, prompt_actions, prompt_rewards,  prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = batch
        prompt_returns_to_go = prompt_returns_to_go[:,:-1,:] 

        prompts = prompt_states, prompt_actions, prompt_rewards,  prompt_returns_to_go, prompt_timesteps, prompt_attention_mask
        if (global_step<= args.warm_train)or(args.zero_shot):
            train_loss = agent.train_step(states, contexts, actions, rewards, dones, rtg, timesteps, masks,None)
            scheduler.step()
        else :
            train_loss = agent.train_step(states, contexts, actions, rewards, dones, rtg, timesteps, masks,prompts)
            scheduler.step()

    

        global_step += 1
        writer.add_scalar('train/loss', train_loss, global_step)
        # pbar.update(1)

  

    
    ### evaluate on five tranining tasks
        if global_step % args.eval_step ==0:
            print(f'\n====== Evaluate at iterations {global_step} =====')
            model.eval()
            avg_epi_return = 0.0
            avg_max_return_offline = 0.0
            print(f'\n---------- Evaluate on five training tasks ----------')


            for task_id in eval_train_task_ids:
                env.reset_task(task_id)
                target_ret = task_info[f'task {task_id}']['return_scale'][1]
                if (global_step<= args.warm_train)or(args.zero_shot):
                    prompt = None
                else:
                    
                    total_rewards = [sum(traj['rewards']) for traj in trajectories_buffer[task_id]]
                            
                    top_indices = sorted(range(len(total_rewards)), key=lambda i: total_rewards[i], reverse=True)[:3]
                    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
                    traj = [trajectories_buffer[task_id][i] for i in top_indices]
                    
                    traj=random.choice(traj)
                    traj = append_error_to_trajectory(world_model,device,context_horizon,traj,args,state_mean, state_std)
                    indices = np.arange(context_horizon, args.max_ep_len - max_len + 1)
                    world_model_error = [traj['errors'][sj : sj+args.max_ep_len].sum() for sj in indices]
                    error_probs = np.array(world_model_error) / np.sum(world_model_error)


                    selected_index = np.random.choice(indices, p=error_probs)

                
                
                    si = selected_index
                 

                    # get sequences from dataset
                    s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                    a.append(traj['actions'][si:si + max_len].reshape(1, -1, action_dim))
                    r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                    timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                    timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                    rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                    if rtg[-1].shape[1] <= s[-1].shape[1]:
                        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                    # padding and state + reward normalization
                    tlen = s[-1].shape[1]
                    # if tlen !=args.K:
                    #     print('tlen not equal to k')
                    s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                    
                    s[-1] = (s[-1] - state_mean) / state_std
                    a[-1] = np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., a[-1]], axis=1)
                    r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                    rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                    timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                    mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
                    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
                    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
                    
                    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
                    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
                    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
                    rtg = rtg[:,:-1,:]
                    rtg = rtg.reshape((1, -1, rtg.shape[-1]))
                    prompt = s, a, r, rtg, timesteps, mask
                
                    
                    
                epi_return, epi_length, traj_per_train = meta_evaluate_episode_rtg(
                    env,
                    state_dim,
                    action_dim,
                    model,
                    context_encoder,
                    max_episode_steps=args.max_episode_steps,
                    scale=args.dt_return_scale,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    target_return=target_ret/args.dt_return_scale,
                    horizon=args.context_horizon,
                    num_eval_episodes=args.num_eval_episodes,
                    prompt=prompt,
                    args = args,
                    epoch = global_step,
                   
                )
                trajectories_buffer[task_id].append(traj_per_train)
                avg_epi_return += epi_return
                avg_max_return_offline += target_ret
                print(f'Evaluate on the {task_id}-th training task, target return {target_ret:.2f}, received return {epi_return:.2f}')

            avg_epi_return /= len(eval_train_task_ids)
            avg_max_return_offline /= len(eval_train_task_ids)
            writer.add_scalars(f'return/train tasks', {'MetaDT':avg_epi_return, 'Oracle':avg_max_return_offline}, global_step)
            
            print(f'\nAverage performance on five training tasks, received return {avg_epi_return:.2f}, average max return from offline dataset {avg_max_return_offline:.2f}')
            
            ### for debugging, print the evaluation trajctory ###
            # print(f'Print the example evaluation trajectory of last evaluation task')
            # env.print_task()
            # for transition in trajectory: print(transition)


        ### evaluate on five test tasks
        if global_step % args.eval_step==0:
            model.eval()
            avg_epi_return = 0.0
            avg_max_return_offline = 0.0
            print(f'\n---------- Evaluate on five test tasks ----------')
            for task_id in test_task_ids:
                env.reset_task(task_id)
                target_ret = task_info[f'task {task_id}']['return_scale'][1]
                if (global_step<= args.warm_train)or(args.zero_shot):
                    prompt = None
                else:
            

                   
                   
                    total_rewards = [sum(traj['rewards']) for traj in trajectories_buffer[task_id]]
                            
                    top_indices = sorted(range(len(total_rewards)), key=lambda i: total_rewards[i], reverse=True)[:3]
                    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
                    traj = [trajectories_buffer[task_id][i] for i in top_indices]
                    
                    traj=random.choice(traj)
                    # print( sum(traj['rewards']))
                    traj = append_error_to_trajectory(world_model,device,context_horizon,traj,args,state_mean, state_std)
                    indices = np.arange(context_horizon, args.max_ep_len - max_len + 1)
                    world_model_error = [traj['errors'][sj : sj+args.max_ep_len].sum() for sj in indices]
                    error_probs = np.array(world_model_error) / np.sum(world_model_error)


                    selected_index = np.random.choice(indices, p=error_probs)

                    si = selected_index
                    
                

                    # get sequences from dataset
                    s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                    a.append(traj['actions'][si:si + max_len].reshape(1, -1, action_dim))
                    r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                    timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                    timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                    rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                    if rtg[-1].shape[1] <= s[-1].shape[1]:
                        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                    # padding and state + reward normalization
                    tlen = s[-1].shape[1]
                    # if tlen !=args.K:
                    #     print('tlen not equal to k')
                    s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                    s[-1] = (s[-1] - state_mean) / state_std
                    a[-1] = np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., a[-1]], axis=1)
                    r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                    rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                    timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                    mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
                    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
                    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
                    
                    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
                    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
                    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
                    rtg = rtg[:,:-1,:]
                    rtg = rtg.reshape((1, -1, rtg.shape[-1]))
                    prompt = s, a, r, rtg, timesteps, mask
                


                    
                
                epi_return, epi_length, traj_per_test = meta_evaluate_episode_rtg(
                    env,
                    state_dim,
                    action_dim,
                    model,
                    context_encoder,
                    max_episode_steps=args.max_episode_steps,
                    scale=args.dt_return_scale,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    target_return=target_ret/args.dt_return_scale,
                    horizon=args.context_horizon,
                    num_eval_episodes=args.num_eval_episodes,
                    prompt=prompt,
                    args = args,
                    epoch=global_step,
                   
                )
                avg_epi_return += epi_return
                avg_max_return_offline += target_ret
                trajectories_buffer[task_id].append(traj_per_test)
                print(f'Evaluate on the {task_id}-th test task, target return {target_ret:.2f}, received return {epi_return:.2f}')

            avg_epi_return /= len(test_task_ids)
            avg_max_return_offline /= len(test_task_ids)
            writer.add_scalars(f'return/test tasks', {'MetaDT':avg_epi_return, 'Oracle':avg_max_return_offline}, global_step)
            
            print(f'\nAverage performance on five test tasks, received return {avg_epi_return:.2f}, average max return from offline dataset {avg_max_return_offline:.2f}')

### for debugging, print the evaluation trajctory ###
# print(f'Print the example evaluation trajectory of last evaluation task')
# env.print_task()
# for transition in trajectory: print(transition)



            print(f'\nElapsed time: {(time.time()-start_time)/60.:.2f} minutes')
# pbar.close() 