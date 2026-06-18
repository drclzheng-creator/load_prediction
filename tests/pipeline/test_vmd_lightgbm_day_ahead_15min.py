from tests.pipeline.pipeline_helpers import (
    run_vmd_lightgbm_pipeline,
)


def test_day_ahead_vmd_lightgbm_pipeline_runs_end_to_end():
    run_vmd_lightgbm_pipeline()
