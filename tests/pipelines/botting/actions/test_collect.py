import cv2


class TestCollectHandler:
    def test_wait_perform_action_collect_not_collected(self, env, data):
        image = cv2.imread(str(data / "test_wait_perform_action_collect_not_collected.png"))
        obs = {"resources": env._filter(env._detect(image))}
        result = env._wait_perform_action(5, obs, obs)  # Collect
        assert result is True  # Should wait, resource have not been collected yet

    def test_wait_perform_action_collect_collected(self, env, data):
        image = cv2.imread(str(data / "test_wait_perform_action_collect_not_collected.png"))
        obs = {"resources": env._filter(env._detect(image))}
        result = env._wait_perform_action(5, obs, obs)  # Collect
        assert result is True  # Should wait, resource have not been collected yet

        # It should wait until resources have been collected for 30 frames
        next_image = cv2.imread(str(data / "test_wait_perform_action_collect_collected.png"))
        next_obs = {"resources": env._filter(env._detect(next_image))}
        for _ in range(30):
            result = env._wait_perform_action(5, obs, next_obs)  # Collect
            assert (
                result is True
            )  # Should wait, resource have been collected but we wait to make sure

        result = env._wait_perform_action(5, obs, next_obs)  # Collect
        assert result is False  # Should not wait, resource have been collected for certain
