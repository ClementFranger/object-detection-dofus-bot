import os
import logging
import gymnasium as gym
import numpy as np
import pyautogui
from gymnasium import spaces
from ultralytics import YOLO
from ultralytics.data.loaders import LoadScreenshots

logger = logging.getLogger(__name__)


class DofusEnv(gym.Env):
    def __init__(self, model, dofus):
        self.model, self.dofus = model, dofus
        # Observation space is the image of the game
        self.height, self.width, self.channels = 1080, 1920, 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, self.channels), dtype=np.uint8
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down" and "collect"
        self.action_space = spaces.Discrete(4)
        self.action_space_mapping = {
            0: self._move_right,
            1: self._move_up,
            2: self._move_left,
            3: self._move_down,
            4: self._collect,
        }

    def _capture_game_screen(self):
        model = YOLO(os.path.join(self.model["path"], "best.pt"))

        loader = LoadScreenshots(
            f"screen 2 {self.dofus.left} {self.dofus.top} {self.dofus.width} {self.dofus.height}"
        )  # Do not forget 'screen' as source
        for screen, img, _ in loader:
            return model.predict(img[0], imgsz=(1920, 1088), verbose=False)

    def _move_right(self):
        pyautogui.click(100, 100)

    def _move_up(self):
        pyautogui.click(100, 100)

    def _move_left(self):
        pyautogui.click(100, 100)

    def _move_down(self):
        pyautogui.click(100, 100)

    def _collect(self):
        pyautogui.click(100, 100)

    def _perform_action(self, action, current_observation):
        logger.info(f"Performing action {self.action_space_mapping[action].__name__}")
        self.action_space_mapping[action]()

    def _compute_reward(self):
        # Compute the reward based on the current game state and action
        # Return the reward
        pass

    def _is_episode_done(self):
        # Check if the episode is done (e.g., if the game is over or the agent achieves its goal)
        # Return True or False
        pass

    def _get_info(self):
        # Check if the episode is done (e.g., if the game is over or the agent achieves its goal)
        # Return True or False
        pass

    def step(self, action):
        # Perform the action in the Dofus game environment
        self._perform_action(action, self._capture_game_screen())

        # Capture the next game screen image as the next observation
        next_observation = self._capture_game_screen()

        # Compute the reward based on the game state and action
        reward = self._compute_reward()

        # Check if the episode is done (e.g., if the game is over or the agent achieves its goal)
        done = self._is_episode_done()

        # Game info
        info = self._get_info()

        # Return the next observation, reward, done flag, and optional information
        return next_observation, reward, done, False, info

    def reset(self):
        # Reset the Dofus game environment to its initial state

        # Perform any necessary initialization steps, such as restarting the game or resetting the agent's position

        # Capture the initial game screen image as the initial observation

        return self._capture_game_screen(), {}
