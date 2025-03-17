import keyboard
import pyautogui
import time

from object_detection_dofus_bot.pipelines.botting import Obs


class MapHandler:
    MOVE_MODE = "keyboard"
    DEFAULT_KEYS = dict(right="d", up="z", left="q", down="s")

    def __init__(self, dofus, **kwargs):
        self.dofus = dofus
        self.MOVE_MODE = kwargs.get("move_mode") or self.MOVE_MODE
        self.DEFAULT_KEYS = kwargs.get("default_keys") or self.DEFAULT_KEYS

    def move_right(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["right"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(
                self.dofus.left + self.dofus.width - 300, self.dofus.top + self.dofus.height / 2
            )

    def move_up(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["up"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(self.dofus.left + self.dofus.width / 2, self.dofus.top + 45)

    def move_left(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["left"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(self.dofus.left + 300, self.dofus.top + self.dofus.height / 2)

    def move_down(self, *args) -> None:
        if self.MOVE_MODE == "keyboard":
            key = self.DEFAULT_KEYS["down"]
            keyboard.press(key)
            time.sleep(0.1)
            keyboard.release(key)
        else:
            pyautogui.click(
                self.dofus.left + self.dofus.width / 2, self.dofus.top + self.dofus.height - 140
            )

    def collect(self, obs: Obs) -> None:
        x1, y1, x2, y2 = map(int, obs["resources"].xyxy[0])
        x, y = (x1 + x2) / 2, (y1 + y2) / 2

        # Since Dofus Unity, a simple click won't do anymore
        # pyautogui.click(x, y)
        pyautogui.mouseDown(x, y)  # Simule l'appui sur le clic gauche
        time.sleep(0.1)  # Pause de 100ms (0.1s)
        pyautogui.mouseUp(x, y)  # Relâche le clic gauche

    def wait_perform_action(self, obs: Obs, next_ops: Obs):
        """Wait map has changed"""
        return obs["map"] == next_ops["map"] if obs["map"] and next_ops["map"] else True
