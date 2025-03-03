import gymnasium as gym
import keyboard
import logging
import numpy as np
import os
import pyautogui
import supervision as sv
import time
from gymnasium import spaces
from supervision import Detections
from ultralytics import YOLO
from ultralytics.data.loaders import LoadScreenshots

logger = logging.getLogger(__name__)


class DofusEnv(gym.Env):
    MOVE_MODE = "keyboard"
    DEFAULT_KEYS = dict(right="d", up="z", left="q", down="s")
    INTERACTIVE_ZONE = dict(
        right=17, up=1, left=17, down=15
    )  # Percentage of total window non interactive for each size

    def __init__(self, model, dofus, **kwargs):
        self.model, self.dofus, self.channels = (
            YOLO(os.path.join(model["path"], "best.pt")),
            dofus,
            3,
        )
        self.MOVE_MODE = kwargs.get("move_mode") or self.MOVE_MODE
        self.DEFAULT_KEYS = kwargs.get("default_keys") or self.DEFAULT_KEYS

        self.screen = LoadScreenshots(
            f"screen 2 {self.dofus.left} {self.dofus.top} {self.dofus.width} {self.dofus.height}"
        )  # Do not forget 'screen' as source

        # Observation space is the image of the game
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.dofus.height, self.dofus.width, self.channels),
            dtype=np.uint8,
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down" and "collect"
        self.action_space = spaces.Discrete(6)
        self.action_space_mapping = {
            0: self._move_right,
            1: self._move_up,
            2: self._move_left,
            3: self._move_down,
            4: self._do_nothing,
            5: self._collect,
        }
        # Last action to be collect
        assert list(self.action_space_mapping.values())[-1].__name__ == "_collect"

    @property
    def interactive_zone(self):
        # Define interactive area
        width, height = self.dofus.width, self.dofus.height
        left_margin = int(width * self.INTERACTIVE_ZONE["left"] / 100)
        right_margin = int(width * (1 - self.INTERACTIVE_ZONE["right"] / 100))
        top_margin = int(height * self.INTERACTIVE_ZONE["up"] / 100)
        bottom_margin = int(height * (1 - self.INTERACTIVE_ZONE["down"] / 100))

        return sv.PolygonZone(
            np.array(
                [
                    [left_margin, top_margin],
                    [right_margin, top_margin],
                    [right_margin, bottom_margin],
                    [left_margin, bottom_margin],
                ]
            )
        )

    def _detect(self, img) -> Detections:
        prediction = self.model.predict(img, imgsz=(1920, 1088), verbose=False)[0]
        detections = sv.Detections.from_ultralytics(prediction)
        return detections

    def _filter(self, detections: Detections) -> Detections:
        mask = self.interactive_zone.trigger(detections=detections)
        filtered_detections = detections[mask]
        logger.info(
            f"Detections before filtering: {len(detections)}, after filtering: {len(filtered_detections)}"
        )
        return filtered_detections

    def _capture_game_screen(self) -> Detections | None:
        for _, img, _ in self.screen:
            detections = self._detect(img[0])
            filtered_detections = self._filter(detections)
            return filtered_detections

    def _do_nothing(self, *args) -> None:
        pass

    def _move_right(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["right"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(
                self.dofus.left + self.dofus.width - 300, self.dofus.top + self.dofus.height / 2
            )

    def _move_up(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["up"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(self.dofus.left + self.dofus.width / 2, self.dofus.top + 45)

    def _move_left(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["left"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(self.dofus.left + 300, self.dofus.top + self.dofus.height / 2)

    def _move_down(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["down"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(
                self.dofus.left + self.dofus.width / 2, self.dofus.top + self.dofus.height - 140
            )

    def _collect(self, current_observation: Detections) -> None:
        x1, y1, x2, y2 = map(int, current_observation.xyxy[0])
        x, y = (x1 + x2) / 2, (y1 + y2) / 2

        # Since Dofus Unity, a simple click won't do anymore
        # pyautogui.click(x, y)
        pyautogui.mouseDown(x, y)  # Simule l'appui sur le clic gauche
        time.sleep(0.1)  # Pause de 100ms (0.1s)
        pyautogui.mouseUp(x, y)  # Relâche le clic gauche

    def _perform_action(self, action: int, current_observation: Detections):
        logger.info(f"Performing action {self.action_space_mapping[action].__name__}")
        self.action_space_mapping[action](current_observation)

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

    def step(self, action, **kwargs):
        # Perform the action in the Dofus game environment
        self._perform_action(action, kwargs.get("current_observation"))
        # Important to wait before next observation (changing map)
        time.sleep(10)

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
