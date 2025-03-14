import cv2
import pytest

from object_detection_dofus_bot.pipelines.botting.agent import (
    DofusFarmAgent,
    DofusCoinBouftouFarmAgent,
)


class TestDofusFarmAgent:
    @pytest.mark.parametrize(
        "image, expected_action",
        [
            ("test_collect", [5]),  # Collect when resources are detected
            (
                "test_do_not_collect_1",
                [0, 1, 2, 3, 4],
            ),  # Do not collect when there are no resources detected
            ("test_do_not_collect_2", [0, 1, 2, 3, 4]),
            # Do not collect when there are no resources detected inside the interactive zone
        ],
    )
    def test_get_action(self, catalog_config, env, data, image, expected_action):
        agent = DofusFarmAgent(env, **catalog_config["parameters"]["agent"])
        image = cv2.imread(str(data / f"{image}.png"))

        obs = {
            "image": image,
            "resources": env._filter(env._detect(image)),
            "in_combat": False,
            "pods": False,
        }

        _, action = agent.get_action(obs)
        assert action in expected_action


class TestDofusCoinBouftouFarmAgent:
    def test_route(self, catalog_config, env, data):
        agent = DofusCoinBouftouFarmAgent(env, **catalog_config["parameters"]["agent"])
        image = cv2.imread(str(data / "test_do_not_collect_1.png"))

        obs = {
            "image": image,
            "resources": env._filter(env._detect(image)),
            "in_combat": False,
            "pods": False,
        }

        assert agent.route == [3, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0]

        for step in range(20):
            expected_action = agent.route[step % len(agent.route)]  # Ensure looping
            _, action = agent.get_action(obs)
            assert (
                action == expected_action
            ), f"Step {step}: Expected {expected_action}, got {action}"
