from tests.replay.replay_helpers import run_lstm_replay


def test_lstm_offline_online_replay_ten_rolling_windows_allclose(tmp_path):
    run_lstm_replay(tmp_path)
