import numpy as np

class Agent:
    def __init__(self, num_states, num_actions, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])

    def train(self, env, num_episodes=5000):
        episode_returns = []
        episode_steps = []

        for ep in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Q-learning update
                best_next = np.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                total_reward += reward
                steps += 1

            episode_returns.append(total_reward)
            episode_steps.append(steps)

        return episode_returns, episode_steps