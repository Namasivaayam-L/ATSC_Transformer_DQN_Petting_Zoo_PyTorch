import numpy as np
from collections import deque

class Memory:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-3, n_step=1, gamma=0.99):
        self.buffer = []
        self.priorities = []
        self.max_size = buffer_size
        self.idx = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque()

    def add(self, state, action, reward, next_state):
        # accumulate n-step info
        self.n_step_buffer.append((state, action, reward, next_state))
        if len(self.n_step_buffer) < self.n_step:
            return
        # compute n-step return
        R = 0
        for i, (_, _, r, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r
        next_state_n = self.n_step_buffer[-1][3]
        state_0, action_0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        aggregated = (state_0, action_0, R, next_state_n)
        # insert into prioritized buffer
        if len(self.buffer) < self.max_size:
            self.buffer.append(aggregated)
            self.priorities.append(max(self.priorities) if self.priorities else 1.0)
        else:
            self.buffer[self.idx] = aggregated
            self.priorities[self.idx] = max(self.priorities)
            self.idx = (self.idx + 1) % self.max_size
        # remove the oldest transition
        self.n_step_buffer.popleft()

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Cannot sample from empty buffer")
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(N, batch_size, p=probs)
        # increment beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states = zip(*(self.buffer[i] for i in indices))
        return states, actions, rewards, next_states, indices, weights

    def update_priorities(self, indices, new_prios):
        for idx, prio in zip(indices, new_prios):
            self.priorities[idx] = prio