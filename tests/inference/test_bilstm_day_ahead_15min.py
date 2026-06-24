from tests.inference.inference_helpers import run_bilstm_request_json_inference


def test_online_inference_bilstm_reads_prediction_request_json():
    run_bilstm_request_json_inference()
