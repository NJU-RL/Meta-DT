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

from configs import args_hopper, args_point_robot
import environments 
from decision_transformer.trainer import DT_Trainer
from decision_transformer.model import DecisionTransformer
from decision_transformer.dataset import DT_Dataset, convert_data_to_trajectories
from decision_transformer.evaluation import evaluate_episode_rtg


parser = argparse.ArgumentParser()
parser.add_argument('--env_type', type=str, default='point_robot')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--task_id', type=int, default=0)
args, rest_args = parser.parse_known_args()
env_type = args.env_type
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
TASK_ID = args.task_id 

if env_type == 'point_robot':
    args = args_point_robot.get_args(rest_args)
    target_returns = [-5, -10]
else:
    raise NotImplementedError


torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=3, suppress=True)

# make env, single_task setting
env = gym.make(args.env_name)
env.seed(args.seed)
env.load_all_tasks(np.load(f'datasets/{args.env_name}/task_goals.npy'))
env.reset_task(TASK_ID)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

results_dir = f'runs/{args.env_name}/dt'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
writer =  SummaryWriter(results_dir)

# load the dataset 
with open(f'datasets/{args.env_name}/dataset_task_{TASK_ID}.pkl', 'rb') as f:
    data = pickle.load(f)
f.close()

# load the task information
with open(f'datasets/{args.env_name}/task_info.json', 'r') as f:
    task_info = json.load(f)
f.close()
max_return_offline = task_info[f'task {TASK_ID}']['return_scale'][1]

trajectories = convert_data_to_trajectories(data)
train_dataset = DT_Dataset(
    trajectories, 
    args.dt_horizon, 
    args.max_episode_steps, 
    args.dt_return_scale, 
    device
)
state_mean, state_std = train_dataset.state_mean, train_dataset.state_std

### save the argument for debugging
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


train_dataloader = DataLoader(train_dataset, batch_size=args.dt_batch_size, shuffle=True)

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=action_dim,
    max_length=args.dt_horizon,
    max_ep_len=args.max_episode_steps,
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
    lambda steps: min((steps+1)/args.dt_warmup_steps, 1)
)

agent = DT_Trainer(model, optimizer)

global_step = 0
for epoch in range(args.dt_num_epochs):
    print(f'\n========== Epoch {epoch+1} ==========')
    for step, batch in enumerate(train_dataloader):
        states, actions, rewards, dones, rtg, timesteps, masks = batch 
        train_loss = agent.train_step(states, actions, rewards, dones, rtg, timesteps, masks)
        scheduler.step()

        global_step += 1
        writer.add_scalar('train/loss', train_loss, global_step)

    model.eval()
    for target_ret in target_returns:
        epi_return, epi_length, trajectory = evaluate_episode_rtg(
            env,
            state_dim,
            action_dim,
            model,
            max_episode_steps=args.max_episode_steps,
            scale=args.dt_return_scale,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            target_return=target_ret/args.dt_return_scale,
            num_eval_episodes=args.num_eval_episodes,
        )

        writer.add_scalars(f'return/target {target_ret}', {'DT':epi_return, 'Oracle':max_return_offline}, epoch+1)
        print(f'\nEvaluation at target {target_ret}: received return {epi_return:.2f}, episode length {epi_length}')

        # print('\n========== Print the evaluation trajectory for debugging ==========')
        # env.print_task()
        # for traj in trajectory: print(traj)


    print(f'\nElapsed time: {(time.time()-start_time)/60.:.2f} minutes')
    






















