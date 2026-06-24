from tests.inference.inference_helpers import run_sklearn_request_json_inference


def test_online_inference_sklearn_reads_prediction_request_json():
    run_sklearn_request_json_inference()
