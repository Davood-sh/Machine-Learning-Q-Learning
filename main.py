import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FlattenObservation
from q_learning import qlearn, rollouts
from policies import RandomPolicy, GreedyPolicy
import matplotlib.pyplot as plt
import numpy as np

def discretize_state(state, bins):
    """Discretize a continuous state into discrete bins."""
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_idx)

def create_bins(num_bins, low, high):
    """Create bins for discretizing continuous spaces, handling infinite values."""
    bins = []
    for l, h in zip(low, high):
        if np.isinf(h):  # Replace infinity with a large number
            h = 1e6
        if np.isinf(l):  # Replace -infinity with a large negative number
            l = -1e6
        bins.append(np.linspace(l, h, num_bins + 1)[1:-1])
    return bins

def main():
    # Initialize the CartPole environment with render mode and flatten the observation space
    env = TimeLimit(FlattenObservation(gym.make("CartPole-v1")), max_episode_steps=500)

    # Inspect properties: make sure it is an MDP!
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    # Create bins for discretizing the continuous observation space
    num_bins = 10  # Number of bins per observation dimension
    bins = create_bins(num_bins, env.observation_space.low, env.observation_space.high)

    # Wrapper to discretize the state
    def discretized_policy(policy):
        def new_policy(state, *args, **kwargs):
            discretized_state = discretize_state(state, bins)
            return policy(discretized_state, *args, **kwargs)
        return new_policy

    # Understand the environment under the random policy
    random_policy = discretized_policy(RandomPolicy(env.action_space.n))
    avg_return = rollouts(env=env, policy=random_policy, gamma=0.95, n_episodes=5)
    print("Avg return with random policy:", avg_return)

    # Learning
    qtable, rewards = qlearn(env=env, alpha0=0.1, gamma=0.95, max_steps=200000, discretize_fn=lambda state: discretize_state(state, bins))

    # Evaluate the learned policy
    greedy_policy = discretized_policy(GreedyPolicy(qtable))
    avg_return = rollouts(env=env, policy=greedy_policy, gamma=0.95, n_episodes=30)
    print(f"Average Return with learned policy: {avg_return}")

    # Render a single episode using the learned policy
    single_episode_return = rollouts(env=env, policy=greedy_policy, gamma=0.95, n_episodes=1)
    print(f"Single Episode Return with learned policy: {single_episode_return}")

    # Plot rewards over episodes
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.show()

if __name__ == "__main__":
    main()