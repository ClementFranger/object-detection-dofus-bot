import easyocr
import gymnasium as gym
import logging
import numpy as np
import os
import re
import supervision as sv
from datetime import datetime, timedelta
from gymnasium import spaces
from supervision import Detections
from ultralytics import YOLO
from ultralytics.data.loaders import LoadScreenshots

from object_detection_dofus_bot.pipelines.botting import Obs
from object_detection_dofus_bot.pipelines.botting.actions.collect import CollectHandler
from object_detection_dofus_bot.pipelines.botting.actions.map import MapHandler

logger = logging.getLogger(__name__)

reader = easyocr.Reader(["fr"])


class DofusEnv(gym.Env):
    MAP_ZONE = dict(x1=0, y1=40, x2=320, y2=100)
    INTERACTIVE_ZONE = dict(
        right=17, up=1, left=17, down=15
    )  # Percentage of total window non interactive for each size

    def __init__(self, model, dofus, **kwargs):
        self.model, self.dofus, self.channels = (
            YOLO(os.path.join(model["path"], "best.pt")),
            dofus,
            3,
        )

        # Initialize actions handlers
        self.map_handler = MapHandler(dofus, **kwargs)
        self.collect_handler = CollectHandler()

        self.resources = kwargs.get("resources", [])
        logger.info(f"Collecting {self.resources if self.resources else 'all'} resources")

        self.tracker = sv.ByteTrack()
        self.source = kwargs.get("source") or "screen 0"
        self.screen = LoadScreenshots(
            f"{self.source} {dofus.left} {dofus.top} {dofus.width} {dofus.height}"
        )  # Do not forget 'screen' as source

        # Observation space is the state of the game
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.dofus.height, self.dofus.width, self.channels),
                    dtype=np.uint8,
                ),
                "resources": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.dofus.height, self.dofus.width, self.channels),
                    dtype=np.uint8,
                ),
                "in_combat": spaces.Discrete(2),
                "pods": spaces.Discrete(2),
                "map": spaces.Box(
                    low=np.array([-100, -100]),  # Minimum possible coordinates
                    high=np.array([50, 60]),  # Maximum possible coordinates (ex: taille du monde)
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down" and "collect"
        self.action_space = spaces.Discrete(6)
        self.action_space_mapping = {
            0: self.map_handler.move_right,
            1: self.map_handler.move_up,
            2: self.map_handler.move_left,
            3: self.map_handler.move_down,
            4: self._do_nothing,
            5: self.collect_handler.collect,
        }
        # Last action to be collect
        assert list(self.action_space_mapping.values())[-1].__name__ == "collect"

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

    def _get_obs(self) -> Obs:
        """Get all observation for our RL agent"""
        for _, img, _ in self.screen:
            image = img[0]
            resources = self._filter(self._detect(image))
            in_combat = False
            pods = False
            map = self._detect_map(image)
            return {
                "image": image,
                "resources": resources,
                "in_combat": in_combat,
                "pods": pods,
                "map": map,
            }

    def _detect_map(self, img) -> tuple[int, int] | None:
        """Detect map coordinates (top left corner)"""
        result = reader.readtext(
            img[
                self.MAP_ZONE["y1"] : self.MAP_ZONE["y2"], self.MAP_ZONE["x1"] : self.MAP_ZONE["x2"]
            ],
            detail=0,
        )
        if result and len(result) >= 2:
            match = re.search(r"([-]?\d+)\s*,\s*([-]?\d+)", result[1])
            if match:
                x, y = map(int, match.groups())
                return x, y

    def _detect(self, img) -> Detections:
        """Detect resources to collect"""
        prediction = self.model.predict(img, imgsz=(1920, 1088), verbose=False)[0]
        detections = sv.Detections.from_ultralytics(prediction)
        return detections

    def _filter(self, detections: Detections) -> Detections:
        """Filter only detections that are inside the interactive game zone (middle)"""
        if self.resources and detections:
            detections = detections[np.isin(detections.data["class_name"], self.resources)]

        # Filter only inside interactive zone
        mask = self.interactive_zone.trigger(detections=detections)
        filtered_detections = detections[mask]
        logger.debug(
            f"Detections before filtering: {len(detections)}, after filtering: {len(filtered_detections)}"
        )
        tracked_filtered_detections = self.tracker.update_with_detections(filtered_detections)
        return tracked_filtered_detections

    def _do_nothing(self, *args) -> None:
        pass

    def _perform_action(self, action: int, obs: Obs):
        logger.info(f"Performing action {self.action_space_mapping[action].__name__}")
        return self.action_space_mapping[action](obs)

    def _wait_perform_action(self, action: int, obs: Obs, next_ops: Obs) -> bool:
        wait = True
        if isinstance(getattr(self.action_space_mapping[action], "__self__", None), MapHandler):
            wait = self.map_handler.wait_perform_action(obs, next_ops)
        if isinstance(getattr(self.action_space_mapping[action], "__self__", None), CollectHandler):
            wait = self.collect_handler.wait_perform_action(obs, next_ops)
        return wait

    def _compute_reward(self):
        # Compute the reward based on the current game state and action
        # Return the reward
        pass

    def _is_episode_done(self):
        # Check if the episode is done (e.g., if the game is over or the agent achieves its goal)
        # Return True or False
        pass

    def _get_info(self):
        pass

    def step(self, action: int, obs: Obs):
        # Perform the action in the Dofus game environment
        self._perform_action(action, obs)

        # Important to wait before next observation (changing map)
        end_time = datetime.now() + timedelta(seconds=10)
        while self._wait_perform_action(action, obs, self._get_obs()) and datetime.now() < end_time:
            logger.debug("Waiting for action to finish")

        # Capture the next game screen image as the next observation
        next_observation = self._get_obs()

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

        return self._get_obs(), {}
