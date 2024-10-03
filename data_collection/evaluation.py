import numpy as np


def evaluate_episode_return(env, agent, num_eval_episodes=10,max_len=200):
    avg_return = 0.
    for _  in range(num_eval_episodes):
        state = env.reset()
        epi_return = 0.
        done = False
        for _ in range(max_len):
        # while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            epi_return += reward
            state = np.copy(next_state)
            if done:
                break

        avg_return += epi_return
    avg_return /= num_eval_episodes
    return avg_return
