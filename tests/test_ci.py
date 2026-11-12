import copy

import numpy as np
import pytest

from src.ci import array_digest, data_gate, macro_f1, model_gate, security_gate
from src.model import InputError, preprocess
import src.model as model_module


def test_preprocess_values_and_flattening():
    image = np.zeros((28, 28), dtype=np.uint8)
    image[0, 1] = 128
    image[27, 27] = 255
    result = preprocess(image)
    assert result.shape == (1, 784)
    assert result.dtype == np.float32
    assert result[0, 0] == 0
    assert result[0, 1] == pytest.approx(128 / 255)
    assert result[0, -1] == 1
    assert np.count_nonzero(result) == 2


def test_no_imputation_nan_is_rejected():
    image = np.zeros((28, 28), dtype=float)
    image[0, 0] = np.nan
    with pytest.raises(InputError, match="NaN"):
        preprocess(image)


def test_load_and_predict_known_artifact(tmp_path, monkeypatch):
    model_path = tmp_path / "model.npz"
    hash_path = tmp_path / "model.sha256"
    weights = np.zeros((10, 784), dtype=np.float32)
    bias = np.zeros(10, dtype=np.float32)
    bias[7] = 1
    np.savez_compressed(model_path, weights=weights, bias=bias, classes=np.arange(10))
    hash_path.write_text(model_module.file_sha256(model_path) + "  model.npz\n")
    monkeypatch.setattr(model_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_module, "MODEL_HASH_PATH", hash_path)
    model = model_module.load_model()
    assert model_module.predict(model, np.zeros((28, 28), dtype=np.uint8)).tolist() == [7]
    model_path.write_bytes(model_path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        model_module.load_model()


def test_macro_f1_known_example():
    # Each class: TP=1, FP=1, FN=1 => precision=recall=F1=0.5.
    assert macro_f1([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)


@pytest.fixture
def sample(tmp_path):
    x = np.zeros((10, 28, 28), dtype=np.uint8)
    y = np.arange(10, dtype=np.uint8)
    policy = {"version": "test-v1", "max_file_bytes": 20000,
              "pixel_dtype": "uint8", "label_dtype": "uint8", "image_shape": [28, 28],
              "min_objects": 10, "max_objects": 10, "min_class_fraction": 0.09,
              "max_class_fraction": 0.11, "images_sha256": array_digest(x),
              "labels_sha256": array_digest(y)}
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, images=x, labels=y, version=np.array("test-v1"))
    return path, policy, x, y


def test_valid_data(sample):
    path, policy, _, _ = sample
    _, _, report = data_gate(path, policy)
    assert report["class_counts"] == [1] * 10


@pytest.mark.parametrize("change", ["shape", "dtype", "labels", "missing", "version", "hash", "distribution", "size"])
def test_bad_data(sample, change):
    path, policy, x, y = sample
    content = {"images": x, "labels": y, "version": np.array("test-v1")}
    if change == "shape":
        content["images"] = x[:, :, :27]
    elif change == "dtype":
        content["images"] = x.astype(float)
        content["images"][0, 0, 0] = np.nan
    elif change == "labels":
        content["labels"][0] = 10
    elif change == "missing":
        del content["labels"]
    elif change == "version":
        content["version"] = np.array("wrong")
    elif change == "hash":
        content["images"][0, 0, 0] = 1
    elif change == "distribution":
        content["labels"] = np.zeros(10, dtype=np.uint8)
    else:
        policy["max_file_bytes"] = 1
    np.savez_compressed(path, **content)
    with pytest.raises(ValueError):
        data_gate(path, policy)


@pytest.fixture
def metrics():
    baseline = {"metric": "f1_macro", "value": 0.8, "dataset_version": "v1",
                "data_identity": "hash", "sample_size": 1000}
    policy = {"allowed_relative_drop": 0.05, "dataset_version": "v1",
              "data_identity": "hash", "sample_size": 1000}
    return baseline, policy


@pytest.mark.parametrize("value,passed", [(0.76, True), (0.759, False), (0.9, True)])
def test_model_gate_threshold(metrics, value, passed):
    baseline, policy = metrics
    candidate = dict(baseline, value=value)
    assert model_gate(candidate, baseline, policy)["passed"] is passed
    assert baseline["value"] == 0.8


@pytest.mark.parametrize("value", [None, "0.9", True, float("nan"), float("inf"), -1, 2])
def test_model_gate_invalid_metric(metrics, value):
    baseline, policy = metrics
    with pytest.raises(ValueError):
        model_gate(dict(baseline, value=value), baseline, policy)


@pytest.mark.parametrize("field", ["metric", "value", "dataset_version", "data_identity", "sample_size"])
def test_model_gate_missing_field(metrics, field):
    baseline, policy = metrics
    candidate = dict(baseline)
    del candidate[field]
    with pytest.raises(ValueError):
        model_gate(candidate, baseline, policy)


def test_model_gate_wrong_data(metrics):
    baseline, policy = metrics
    with pytest.raises(ValueError):
        model_gate(dict(baseline, dataset_version="other"), baseline, policy)


def test_model_gate_zero_reference(metrics):
    baseline, policy = metrics
    with pytest.raises(ValueError, match="Zero"):
        model_gate(baseline, dict(baseline, value=0), policy)


def test_security_report_policy():
    clean = {"SchemaVersion": 2, "Metadata": {}, "Results": [{"Target": "test"}]}
    assert security_gate(clean)["passed"]
    vulnerable = copy.deepcopy(clean)
    vulnerable["Results"][0]["Vulnerabilities"] = [{"Severity": "CRITICAL"}]
    assert not security_gate(vulnerable)["passed"]
    vulnerable["Results"][0]["Vulnerabilities"][0]["Severity"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown"):
        security_gate(vulnerable)


@pytest.mark.parametrize("report", [{}, {"SchemaVersion": 2}, {"SchemaVersion": 2, "Metadata": {}, "Results": []}])
def test_missing_scan_fails(report):
    with pytest.raises(ValueError):
        security_gate(report)
