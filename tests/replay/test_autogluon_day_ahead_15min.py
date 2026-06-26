from tests.replay.replay_helpers import run_autogluon_replay


def test_autogluon_offline_online_replay_ten_rolling_windows_allclose(tmp_path):
    run_autogluon_replay(tmp_path)
