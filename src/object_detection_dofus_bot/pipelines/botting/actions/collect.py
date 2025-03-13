import pyautogui
import time

from object_detection_dofus_bot.pipelines.botting import Obs
from object_detection_dofus_bot.pipelines.botting.actions import FactoryInstance


class CollectHandler(FactoryInstance):
    def collect(self, obs: Obs) -> None:
        x1, y1, x2, y2 = map(int, obs["resources"].xyxy[0])
        x, y = (x1 + x2) / 2, (y1 + y2) / 2

        # Since Dofus Unity, a simple click won't do anymore
        # pyautogui.click(x, y)
        pyautogui.mouseDown(x, y)  # Simule l'appui sur le clic gauche
        time.sleep(0.1)  # Pause de 100ms (0.1s)
        pyautogui.mouseUp(x, y)  # Relâche le clic gauche

    def wait_perform_action(self, obs: Obs, next_ops: Obs):
        print(obs["resources"])
        print(obs["resources"].xyxy[0])
        print(next_ops["resources"].xyxy[0])
        return obs["resources"].xyxy[0] == next_ops["resources"].xyxy[0]
