import cv2
from gymnasium import spaces

from object_detection_dofus_bot.pipelines.botting.env import DofusEnv


class TestDofusEnv:
    # def test_move_mode(self, model, dofus):
    #     env = DofusEnv(model, dofus)
    #     assert env.MOVE_MODE == "keyboard"
    #
    #     env = DofusEnv(model, dofus, move_mode="mouse")
    #     assert env.MOVE_MODE == "mouse"
    #
    # def test_default_keys(self, model, dofus):
    #     env = DofusEnv(model, dofus)
    #     assert env.DEFAULT_KEYS == dict(right="d", up="z", left="q", down="s")
    #
    #     env = DofusEnv(model, dofus, default_keys=dict(right="d", up="w", left="a", down="s"))
    #     assert env.DEFAULT_KEYS == dict(right="d", up="w", left="a", down="s")

    def test_actions(self, env):
        assert env.action_space.n == 6
        assert env.action_space_mapping[env.action_space.n - 1] == env.collect_handler.collect

    def test_observations(self, env, dofus):
        assert isinstance(env.observation_space, spaces.Dict)

        assert isinstance(env.observation_space["image"], spaces.Box)
        assert env.observation_space["image"].shape == (dofus.height, dofus.width, env.channels)
        assert isinstance(env.observation_space["resources"], spaces.Box)
        assert isinstance(env.observation_space["in_combat"], spaces.Discrete)
        assert isinstance(env.observation_space["pods"], spaces.Discrete)

    def test_detect(self, env, data):
        image = cv2.imread(str(data / "test_predict.png"))
        assert len(env._detect(image)) == 2

    def test_detect_map(self, env, data):
        image = cv2.imread(str(data / "test_predict.png"))
        assert env._detect_map(image) == (5, 8)

        image = cv2.imread(str(data / "test_filter.png"))
        assert env._detect_map(image) == (1, 8)

        image = cv2.imread(str(data / "test_do_not_collect_1.png"))
        assert env._detect_map(image) == (3, 9)

    def test_filter(self, model, dofus, env, data):
        image = cv2.imread(str(data / "test_filter.png"))

        detections = env._detect(image)
        filtered_detections = env._filter(detections)
        assert len(detections) == 8
        assert len(filtered_detections) == 6

        env = DofusEnv(model, dofus, resources=["frene"])

        detections = env._detect(image)
        filtered_detections = env._filter(detections)

        assert len(detections) == 8
        assert len(filtered_detections) == 5

    def test_interactive_zone(self, env, data):
        image = cv2.imread(str(data / "test_interactive_zone.png"))
        detections = env._detect(image)
        assert len(detections) == 6

        filtered_detections = env._filter(detections)
        assert len(filtered_detections) == 4
