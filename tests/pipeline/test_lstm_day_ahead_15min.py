from tests.pipeline.pipeline_helpers import (
    run_lstm_config_params,
    run_lstm_pipeline,
    run_lstm_pipeline_with_parameter_tuning,
)


def test_day_ahead_lstm_pipeline_runs_end_to_end():
    run_lstm_pipeline()


def test_day_ahead_lstm_pipeline_runs_end_to_end_with_parameterTuning():
    run_lstm_pipeline_with_parameter_tuning(
        enable_parameter_tuning=True,
        n_trials=2,
    )


def test_lstm_from_config_accepts_sampling_and_point_head_params():
    run_lstm_config_params()
