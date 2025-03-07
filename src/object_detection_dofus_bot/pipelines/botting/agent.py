from collections import defaultdict

import numpy as np

from object_detection_dofus_bot.pipelines.botting import Obs


class DofusAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DofusFarmAgent(DofusAgent):
    def get_action(self, obs: Obs):
        """
        Returns the collect action first until not ressource is available
        otherswise a random action with probability epsilon to ensure exploration.
        """
        # collect whenever possible
        if obs["resources"]:
            return obs, self.env.action_space.n - 1

        # return a random action to explore the environment (except collect)
        else:
            mask = np.ones(self.env.action_space.n, dtype=np.int8)  # All actions are valid
            mask[self.env.action_space.n - 1] = 0  # Set the "collect" action to 0
            return obs, self.env.action_space.sample(mask=mask)


class DofusCoinBouftouFarmAgent(DofusAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = [3, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0]
        self.current_step = 0
        self.collect = 0

    def get_action(self, obs: Obs):
        """
        Returns the collect action first until not ressource is available
        otherwise the specified route will be followed
        """
        # collect whenever possible
        if obs["resources"]:
            self.collect += 1
            return obs, self.env.action_space.n - 1

        # Follow the specified route
        else:
            action = self.route[self.current_step]
            self.current_step = (self.current_step + 1) % len(self.route)  # Boucle sur la route
            self.collect = 0
            return obs, action
