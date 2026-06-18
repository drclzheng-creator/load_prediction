from tests.pipeline.pipeline_helpers import (
    run_cnn_lstm_pipeline,
)


def test_day_ahead_cnn_lstm_pipeline_runs_end_to_end():
    run_cnn_lstm_pipeline()
