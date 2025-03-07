import logging
import pyautogui
import time

from object_detection_dofus_bot.pipelines.botting import Obs

logger = logging.getLogger(__name__)


class CollectHandler:
    def __init__(self):
        self.attempts = 0

    def collect(self, obs: Obs):
        x1, y1, x2, y2 = map(int, obs["resources"].xyxy[0])
        x, y = (x1 + x2) / 2, (y1 + y2) / 2

        # Since Dofus Unity, a simple click won't do anymore
        # pyautogui.click(x, y)
        pyautogui.mouseDown(x, y)  # Simule l'appui sur le clic gauche
        time.sleep(0.1)  # Pause de 100ms (0.1s)
        pyautogui.mouseUp(x, y)  # Relâche le clic gaucheq

    def wait_perform_action(self, obs: Obs, next_ops: Obs):
        """Wait until tracked resource is collected"""
        # If the resources have been collected, then it is not in next_ops detection
        # (or next_ops detection is empty because no more resources)
        if obs["resources"].tracker_id[0] not in next_ops["resources"].tracker_id:
            self.attempts += 1
            logger.debug(f"Resource seems to be collected, incrementing attempts : {self.attempts}")
        else:
            self.attempts = 0
        if self.attempts > 30:
            logger.debug("Resource have definitely been collected, resetting attempts to 0")
            self.attempts = 0
            return False
        return True
