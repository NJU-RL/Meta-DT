import numpy as np 
import torch 

def evaluate_episode(env, agent, context_encoder, horizon=4, episodes=10, device='cpu'):
    
    avg_epi_return = 0.
    for _ in range(episodes):
        state = env.reset()
        epi_return = 0.
        states_traj = np.zeros((env._max_episode_steps, env.observation_space.shape[0]))
        actions_traj = np.zeros((env._max_episode_steps, env.action_space.shape[0]))
        rewards_traj = np.zeros((env._max_episode_steps, 1))

        for step in range(env._max_episode_steps):
            if step == 0:
                state_seg = np.zeros((horizon, env.observation_space.shape[0]))
                action_seg = np.zeros((horizon, env.action_space.shape[0]))
                reward_seg = np.zeros((horizon, 1))
            else:
                start_idx = max(0, step-horizon)
                state_seg = states_traj[start_idx : step]
                action_seg = actions_traj[start_idx : step]
                reward_seg = rewards_traj[start_idx : step]

            length_gap = horizon - state_seg.shape[0] 
            state_seg = np.pad(state_seg, ((length_gap, 0),(0, 0)))
            action_seg = np.pad(action_seg, ((length_gap, 0),(0, 0)))
            reward_seg = np.pad(reward_seg, ((length_gap, 0),(0, 0)))

            state_seg = torch.FloatTensor(state_seg).to(device).unsqueeze(1)
            action_seg = torch.FloatTensor(action_seg).to(device).unsqueeze(1)
            reward_seg = torch.FloatTensor(reward_seg).to(device).unsqueeze(1)

            context = context_encoder(state_seg, action_seg, reward_seg).detach()
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)

            _, _, action_mean = agent.policy.sample(state_tensor, context)
            action = action_mean.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            epi_return += reward

            states_traj[step] = np.copy(state)
            actions_traj[step] = np.copy(action)
            rewards_traj[step] = np.copy(reward)
            
            state = np.copy(next_state)
            if done:
                break 
        
        trajectory = np.concatenate([states_traj, actions_traj, rewards_traj], axis=-1)
        avg_epi_return += epi_return

    avg_epi_return /= episodes

    return avg_epi_return, trajectory