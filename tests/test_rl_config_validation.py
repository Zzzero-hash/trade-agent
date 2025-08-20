from hydra import compose, initialize

from trade_agent.config.rl_structured import validate_rl_config


def test_validate_rl_default_config() -> None:
    # config_path is relative to this test file directory
    # so we use '../conf/rl'
    with initialize(version_base=None, config_path="../conf/rl"):
        cfg = compose(config_name="config")
        validated = validate_rl_config(cfg)
        assert validated.algo in {"ppo", "sac"}
        if validated.algo == "ppo":
            assert validated.ppo is not None
        if validated.algo == "sac":
            assert validated.sac is not None
        assert validated.env.window_size > 0
        assert validated.training.total_timesteps > 0
