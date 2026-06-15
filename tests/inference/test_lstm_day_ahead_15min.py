from tests.inference.inference_helpers import (
    run_lstm_manifest_requires_context_plus_lag_history,
    run_lstm_saved_model_inference,
)


def test_lstm_manifest_requires_context_plus_lag_history():
    run_lstm_manifest_requires_context_plus_lag_history()


def test_online_inference_lstm_saved_model():
    run_lstm_saved_model_inference()
