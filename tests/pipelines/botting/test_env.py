import cv2

from object_detection_dofus_bot.pipelines.botting.env import DofusEnv


class TestDofusEnv:
    def test_move_mode(self, model, dofus):
        env = DofusEnv(model, dofus)
        assert env.MOVE_MODE == "keyboard"

        env = DofusEnv(model, dofus, move_mode="mouse")
        assert env.MOVE_MODE == "mouse"

    def test_default_keys(self, model, dofus):
        env = DofusEnv(model, dofus)
        assert env.DEFAULT_KEYS == dict(right="d", up="z", left="q", down="s")

        env = DofusEnv(model, dofus, default_keys=dict(right="d", up="w", left="a", down="s"))
        assert env.DEFAULT_KEYS == dict(right="d", up="w", left="a", down="s")

    def test_observations(self, env):
        assert env.action_space.n == 6

    def test_actions(self, env, dofus):
        assert env.observation_space.shape == (dofus.height, dofus.width, env.channels)

    def test_detect(self, env, data):
        image = cv2.imread(str(data / "test_predict.png"))
        assert len(env._detect(image)) == 2

    def test_interactive_zone(self, env, data):
        image = cv2.imread(str(data / "test_interactive_zone.png"))
        detections = env._detect(image)
        assert len(detections) == 6

        filtered_detections = env._filter(detections)
        assert len(filtered_detections) == 4
