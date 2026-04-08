import numpy as np

# Reinforcement Learning Agent for GA Parameter Tuning
class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.epsilon = 0.05
        self.gamma = 0.95
        self.q_table = np.zeros((state_size, action_size))
    
    def get_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q