from tests.pipeline.pipeline_helpers import (
    run_lightgbm_pipeline,
    run_lightgbm_pipeline_with_parameter_tuning,
)


def test_day_ahead_lightgbm_pipeline_runs_end_to_end():
    run_lightgbm_pipeline()


def test_day_ahead_lightgbm_pipeline_runs_end_to_end_with_parameterTuning():
    run_lightgbm_pipeline_with_parameter_tuning(
        enable_parameter_tuning=True,
        n_trials=2,
    )
