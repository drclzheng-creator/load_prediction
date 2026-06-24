from tests.inference.inference_helpers import run_autogluon_request_json_inference


def test_online_inference_autogluon_reads_prediction_request_json():
    run_autogluon_request_json_inference()
