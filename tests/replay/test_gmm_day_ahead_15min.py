from tests.replay.replay_helpers import run_gmm_replay


def test_gmm_offline_online_replay_ten_rolling_windows_allclose(tmp_path):
    run_gmm_replay(tmp_path)
