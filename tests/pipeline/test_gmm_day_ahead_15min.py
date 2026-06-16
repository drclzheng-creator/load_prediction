from tests.pipeline.pipeline_helpers import (
    run_gmm_pipeline_with_distribution_plots,
    run_gmm_pipeline_with_parameter_tuning,
)


def test_day_ahead_gmm_pipeline_runs_end_to_end_with_distribution_plots():
    run_gmm_pipeline_with_distribution_plots()


def test_day_ahead_gmm_pipeline_runs_end_to_end_with_parameterTuning():
    run_gmm_pipeline_with_parameter_tuning(
        enable_parameter_tuning=True,
        n_trials=2,
    )
