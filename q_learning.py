import numpy as np
from collections import defaultdict
from gymnasium import Env
from policies import EpsGreedyPolicy

# Algorithm
def qlearn(env: Env, alpha0: float, gamma: float, max_steps: int, discretize_fn, epsilon_max: float = 1.0, epsilon_min: float = 0.01, lambda_decay: float = 0.001):
    """Q-learning training loop. Returns estimated optimal Q-table."""
    #Q-Table Initialization
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = EpsGreedyPolicy(Q)

    done = True
    episode_rewards = []
    current_episode_reward = 0
    #Training Loop
    for step in range(max_steps):
        #Episode Start
        #When done is True, it means the previous episode has ended.
        if done:
            episode_rewards.append(current_episode_reward)
            obs, _ = env.reset()  # get the observation from the reset
            obs = discretize_fn(obs)
            current_episode_reward = 0

        # Exponential decay for epsilon
        eps = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-lambda_decay * step)
        #policy(obs, eps): Selects an action based on the current state obs and the epsilon value eps
        action = policy(obs, eps)
        obs2, rew, done, truncated, _ = env.step(action)
        done = done or truncated
        #Discretize Next State
        obs2 = discretize_fn(obs2)
        Q[obs][action] += alpha0 * (rew + gamma * np.max(Q[obs2]) - Q[obs][action])
        obs = obs2
        current_episode_reward += rew

    return Q, episode_rewards

def rollouts(env: Env, policy, gamma: float, n_episodes: int, render=False) -> float:
    """Perform rollouts and compute the average discounted return."""
    sum_returns = 0.0

    done = False
    obs, _ = env.reset()  # get the observation from the reset
    discounting = 1
    ep = 0

    while True:
        if done:
            if render:
                print("New episode")
            obs, _ = env.reset()  # get the observation from the reset
            discounting = 1
            ep += 1
            if ep >= n_episodes:
                break

        action = policy(obs)
        obs, rew, done, truncated, _ = env.step(action)
        done = done or truncated
        sum_returns += rew * discounting
        discounting *= gamma

        if render:
            print(env.render())  # Print the rendering output to the console

    return sum_returns / n_episodes
