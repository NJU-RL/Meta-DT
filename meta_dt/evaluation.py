import numpy as np
import torch
from collections import OrderedDict


def meta_evaluate_episode_rtg(
        env,
        state_dim,
        action_dim,
        model,
        context_encoder,
        max_episode_steps=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        horizon=4,
        context_dim=16,
        num_eval_episodes=10,
        prompt=None,
        args =None,
        epoch = 0,
        ):

    model.eval(); context_encoder.eval()
    model.to(device=device); context_encoder.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    avg_epi_return = 0.
    avg_epi_len = 0
    for _ in range(num_eval_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        contexts = torch.zeros((1, context_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        states_traj = np.zeros((args.max_episode_steps, env.observation_space.shape[0]))
        actions_traj = np.zeros((args.max_episode_steps, env.action_space.shape[0]))
        rewards_traj = np.zeros((args.max_episode_steps, 1))

        target_returns = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        for t in range(max_episode_steps):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                contexts.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_returns.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt,
                args = args,
                epoch = epoch
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            step_result = env.step(action)
            state = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            states_traj[t] = np.copy(states[-1].detach().cpu().numpy().reshape(-1))
            actions_traj[t] = np.copy(action)
            rewards_traj[t] = np.copy(reward)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            # compute the current context
            state_seg = states_traj[t+1-horizon : t+1]
            action_seg = actions_traj[t+1-horizon : t+1]
            reward_seg = rewards_traj[t+1-horizon : t+1]
            next_state_seg = states_traj[t+2-horizon : t+2]

            length_gap = horizon - state_seg.shape[0] 
            state_seg = np.pad(state_seg, ((length_gap, 0),(0, 0)))
            action_seg = np.pad(action_seg, ((length_gap, 0),(0, 0)))
            reward_seg = np.pad(reward_seg, ((length_gap, 0),(0, 0)))
            next_state_seg = np.pad(next_state_seg, ((length_gap, 0),(0, 0)))
            

            state_seg = torch.FloatTensor(state_seg).to(device).unsqueeze(1)
            action_seg = torch.FloatTensor(action_seg).to(device).unsqueeze(1)
            reward_seg = torch.FloatTensor(reward_seg).to(device).unsqueeze(1)
            next_state_seg = torch.FloatTensor(next_state_seg).to(device).unsqueeze(1)
            if args.env_name == 'WalkerRandParams-v0':
                # cur_context = context_encoder(state_seg, action_seg, reward_seg,next_state_seg).detach().reshape(1, -1)
                cur_context = context_encoder(state_seg, action_seg, reward_seg).detach().reshape(1, -1)
            else:
                cur_context = context_encoder(state_seg, action_seg, reward_seg).detach().reshape(1, -1)
            contexts = torch.cat([contexts, cur_context], dim=0)
            if args.env_name=='Reach':
                reward = torch.from_numpy(reward).type(torch.cuda.FloatTensor)
                
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_returns[0,-1] - (reward/scale)
            else:
                pred_return = target_returns[0,-1]
            target_returns = torch.cat(
                [target_returns, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            avg_epi_return += reward
            avg_epi_len += 1

            if done:
                break
        

        # trajectory = torch.cat((states[:-1], actions, rewards.reshape(-1,1)), dim=-1).detach().cpu().numpy()
    trajectory = OrderedDict([
    ('observations', states[:-1].cpu().detach().numpy()),
    ('actions', actions.cpu().detach().numpy()),
    ('rewards', rewards.cpu().detach().numpy()),
    ('next_observations', states[1:].cpu().detach().numpy()),
])

    return avg_epi_return/num_eval_episodes, avg_epi_len/num_eval_episodes, trajectory
