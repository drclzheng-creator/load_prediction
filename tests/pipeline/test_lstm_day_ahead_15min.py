from tests.pipeline.pipeline_helpers import (
    run_lstm_config_params,
    run_lstm_pipeline,
)


def test_day_ahead_lstm_pipeline_runs_end_to_end():
    run_lstm_pipeline()


def test_lstm_from_config_accepts_sampling_and_point_head_params():
    run_lstm_config_params()

