import numpy as np
import torch



def evaluate_episode_rtg(
        env,
        state_dim,
        action_dim,
        model,
        max_episode_steps=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        num_eval_episodes=10,
        ):

    model.eval()
    model.to(device=device)

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
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        target_returns = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        for t in range(max_episode_steps):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_returns.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            step_result = env.step(action)
            state = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
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
        
        trajectory = torch.cat((states[:-1], actions, rewards.reshape(-1,1)), dim=-1).detach().cpu().numpy()

    return avg_epi_return/num_eval_episodes, avg_epi_len/num_eval_episodes, trajectory