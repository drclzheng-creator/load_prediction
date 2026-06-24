from tests.inference.inference_helpers import run_lightgbm_request_json_inference


def test_online_inference_lightgbm_reads_prediction_request_json():
    run_lightgbm_request_json_inference()
