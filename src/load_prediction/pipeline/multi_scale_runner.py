"""Multi-profile pipeline runner."""

from __future__ import annotations

from load_prediction.configs import PipelineConfig, forecast_profile_preset
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.pipeline.pipeline_result import PipelineRunResult


class MultiScaleForecastRunner:
    """Run the same dataset through multiple forecast-profile-specific pipelines."""

    def __init__(self, base_config: PipelineConfig, scale_names: list[str]):
        self.base_config = base_config
        self.scale_names = scale_names

    def run(self, data: TimeSeriesDataset) -> dict[str, PipelineRunResult]:
        from load_prediction.pipeline.forecast_pipeline import LoadForecastPipeline

        results: dict[str, PipelineRunResult] = {}
        for name in self.scale_names:
            scale = forecast_profile_preset(name)
            config = PipelineConfig(
                data=self.base_config.data,
                cleaning=self.base_config.cleaning,
                model=self.base_config.model,
                evaluation=self.base_config.evaluation,
                postprocess=self.base_config.postprocess,
                output=self.base_config.output,
                scale=scale,
            )
            results[name] = LoadForecastPipeline(config).run(data)
        return results

