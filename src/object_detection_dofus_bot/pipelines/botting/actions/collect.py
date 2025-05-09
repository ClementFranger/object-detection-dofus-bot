import logging
import pyautogui
import time

from object_detection_dofus_bot.pipelines.botting import Obs

logger = logging.getLogger(__name__)


class CollectHandler:
    def __init__(self):
        # self.attempts = 0
        self.start_time = None

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
            if not self.start_time:
                self.start_time = time.time()
                # self.attempts += 1
                logger.info("Resource seems to be collected, launching timer")
            print(f"elapsed time : {time.time() - self.start_time}")
        else:
            # self.attempts = 0
            self.start_time = None
            logger.debug("Resource is still present resetting timer")
        if self.start_time and time.time() - self.start_time > 5:
            logger.info("Resource have definitely been collected, resetting timer to 0")
            self.start_time = None
            return False
        return True
