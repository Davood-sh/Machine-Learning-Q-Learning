import random
import numpy as np

# Let's define some policies
class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, obs) -> int:
        return random.randint(0, self.n_actions - 1)

class GreedyPolicy:
    def __init__(self, Q):
        self.Q = Q

    def __call__(self, obs) -> int:
        return np.argmax(self.Q[obs])

class EpsGreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
        self.n_actions = len(Q[0])

    def __call__(self, obs, eps: float) -> int:
        greedy = random.random() > eps
        if greedy:
            return np.argmax(self.Q[obs])
        else:
            return random.randint(0, self.n_actions - 1)