from tests.pipeline.pipeline_helpers import (
    run_bilstm_pipeline,
    run_bilstm_pipeline_with_parameter_tuning,
)


def test_day_ahead_bilstm_pipeline_runs_end_to_end():
    run_bilstm_pipeline()


def test_day_ahead_bilstm_pipeline_runs_end_to_end_with_parameterTuning():
    run_bilstm_pipeline_with_parameter_tuning(
        enable_parameter_tuning=True,
        n_trials=2,
    )
