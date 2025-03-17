from object_detection_dofus_bot.pipelines.botting.actions.map import MapHandler


class TestMapHandler:
    def test_move_mode(self, dofus):
        map_handler = MapHandler(dofus)
        assert map_handler.MOVE_MODE == "keyboard"

        map_handler = MapHandler(dofus, move_mode="mouse")
        assert map_handler.MOVE_MODE == "mouse"

    def test_default_keys(self, dofus):
        map_handler = MapHandler(dofus)
        assert map_handler.DEFAULT_KEYS == dict(right="d", up="z", left="q", down="s")

        map_handler = MapHandler(dofus, default_keys=dict(right="d", up="w", left="a", down="s"))
        assert map_handler.DEFAULT_KEYS == dict(right="d", up="w", left="a", down="s")

    def test_wait_perform_action_map_change(self, env):
        obs = {"map": (1, 1)}
        next_obs = {"map": (2, 1)}  # Simulate map change
        result = env._wait_perform_action(0, obs, next_obs)  # Moving right
        assert result is False  # Should not wait, map changed

    def test_wait_perform_action_map_not_change(self, env):
        obs = {"map": (1, 1)}
        result = env._wait_perform_action(0, obs, obs)  # Moving right
        assert result is True  # Should wait, map has not yet changed

        next_obs = {"map": (2, 1)}  # Simulate map change
        result = env._wait_perform_action(0, obs, next_obs)  # Moving right
        assert result is False  # Should wait, map has not yet changed
