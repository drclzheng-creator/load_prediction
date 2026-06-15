from tests.replay.replay_helpers import run_lightgbm_replay


def test_lightgbm_offline_online_replay_ten_rolling_windows_allclose(tmp_path):
    run_lightgbm_replay(tmp_path)
