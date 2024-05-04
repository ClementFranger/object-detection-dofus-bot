import time

import os

from ultralytics import YOLO
from ultralytics.data.loaders import LoadScreenshots

from object_detection_dofus_bot.pipelines.botting.agent import DofusAgent
from object_detection_dofus_bot.pipelines.botting.dofus import DofusEnv


def infer(model, dofus):
    model = YOLO(os.path.join(model["path"], "best.pt"))

    loader = LoadScreenshots(
        f"screen 2 {dofus.left} {dofus.top} {dofus.width} {dofus.height}"
    )  # Do not forget 'screen' as source
    for screen, img, _ in loader:
        model.predict(img[0], show=True, imgsz=(1920, 1088))


def bot(model, dofus, params):
    env = DofusEnv(model, dofus)
    agent = DofusAgent(env, **params)

    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(10)

        # update the agent
        # agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
