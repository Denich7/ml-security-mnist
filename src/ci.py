"""Lab 3: immutable data, metrics and security gates. No training in CI."""
import argparse
import hashlib
import io
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.model import ROOT, file_sha256, load_model, predict

CI = ROOT / "ci"
REPORTS = ROOT / "reports" / "lab3"
SAMPLE = ROOT / "data" / "ci_sample.npz"
SOURCE = ROOT / "data" / "mnist.npz"
SOURCE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SOURCE_SHA = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
VERSION = "mnist-official-test-first-100-per-class-v1"


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def array_digest(value):
    # Canonical NPY bytes include dtype and shape; do not hash NPZ ZIP timestamps.
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def prepare_data():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temp = SOURCE.with_suffix(".download")
        # Fixed HTTPS official source; no user-controlled URL. Hard limit 16 MiB.
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # nosec B310
            payload = response.read(16 * 1024 * 1024 + 1)
        require(len(payload) <= 16 * 1024 * 1024, "MNIST download exceeds limit")
        temp.write_bytes(payload)
        require(file_sha256(temp) == SOURCE_SHA, "MNIST download hash mismatch")
        temp.replace(SOURCE)
    require(file_sha256(SOURCE) == SOURCE_SHA, "MNIST differs from Lab 1")
    with np.load(SOURCE, allow_pickle=False) as archive:
        images, labels = archive["x_test"], archive["y_test"]
        require(images.shape == (10000, 28, 28), "Invalid source images")
        require(labels.shape == (10000,), "Invalid source labels")
        indices = np.concatenate([np.flatnonzero(labels == i)[:100] for i in range(10)])
        np.savez_compressed(SAMPLE, images=images[indices], labels=labels[indices],
                            version=np.array(VERSION))
    return SAMPLE


def data_gate(path, policy):
    require(Path(path).stat().st_size <= policy["max_file_bytes"], "Data file too large")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"images", "labels", "version"}, "Invalid fields")
        x, y, version = archive["images"], archive["labels"], archive["version"].item()
    require(version == policy["version"], "Data version mismatch")
    require(x.dtype == np.dtype(policy["pixel_dtype"]), "Invalid pixel dtype")
    require(y.dtype == np.dtype(policy["label_dtype"]), "Invalid label dtype")
    require(x.ndim == 3 and x.shape[1:] == tuple(policy["image_shape"]), "Invalid image shape")
    require(y.shape == (len(x),), "Invalid labels shape")
    require(policy["min_objects"] <= len(x) <= policy["max_objects"], "Invalid sample size")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "NaN/Inf")
    require(np.all((x >= 0) & (x <= 255)), "Invalid pixel range")
    require(np.isin(y, np.arange(10)).all(), "Invalid labels")
    counts = np.bincount(y, minlength=10)
    proportions = counts / len(y)
    require(np.all((proportions >= policy["min_class_fraction"]) &
                   (proportions <= policy["max_class_fraction"])), "Class distribution violation")
    require(array_digest(x) == policy["images_sha256"], "Images SHA256 mismatch")
    require(array_digest(y) == policy["labels_sha256"], "Labels SHA256 mismatch")
    return x, y, {"version": version, "size": len(y), "class_counts": counts.tolist(),
                  "images_sha256": array_digest(x), "labels_sha256": array_digest(y)}


def macro_f1(truth, predicted):
    # Exactly the Lab 2 macro definition. All 10 classes are present in the CI sample.
    return float(f1_score(truth, predicted, average="macro", zero_division=0))


def evaluate(model, images, labels):
    result = np.concatenate([predict(model, images[i:i + 64]) for i in range(0, len(images), 64)])
    return macro_f1(labels, result)


def model_gate(candidate, baseline, policy):
    for obj in (baseline, candidate):
        require(obj.get("metric") == "f1_macro", "Missing or invalid metric name")
        require(numeric(obj.get("value")) and 0 <= obj["value"] <= 1, "Invalid metric value")
        require(obj.get("dataset_version") == policy["dataset_version"], "Dataset version mismatch")
        require(obj.get("data_identity") == policy["data_identity"], "Dataset hash mismatch")
        require(obj.get("sample_size") == policy["sample_size"], "Sample size mismatch")
    require(baseline["value"] > 0, "Zero baseline requires an explicit absolute policy")
    require(numeric(policy["allowed_relative_drop"]) and
            0 <= policy["allowed_relative_drop"] < 1, "Invalid threshold policy")
    threshold = baseline["value"] * (1 - policy["allowed_relative_drop"])
    return {"candidate": candidate["value"], "baseline": baseline["value"],
            "threshold": threshold, "passed": candidate["value"] >= threshold}


def security_gate(report):
    require(isinstance(report, dict) and report.get("SchemaVersion") == 2, "Invalid Trivy report")
    require(isinstance(report.get("Results"), list) and report["Results"], "Missing scan results")
    require(isinstance(report.get("Metadata"), dict), "Missing scan metadata")
    critical = []
    for result in report["Results"]:
        require(isinstance(result, dict) and "Target" in result, "Invalid scan target")
        vulnerabilities = result.get("Vulnerabilities") or []
        require(isinstance(vulnerabilities, list), "Invalid vulnerability list")
        for item in vulnerabilities:
            require(isinstance(item, dict), "Invalid vulnerability entry")
            severity = item.get("Severity")
            require(severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}, "Unknown severity")
            if severity == "CRITICAL":
                critical.append({key: item.get(key) for key in
                                 ("VulnerabilityID", "PkgName", "InstalledVersion", "FixedVersion")})
    return {"passed": not critical, "critical": critical}


def scenario():
    value = read_json(CI / "scenario.json")["scenario"]
    require(value in {"normal", "data", "model", "security"}, "Unknown demo scenario")
    return value


def initialize():
    # One-time local initialization: never called by the workflow.
    for name in ("data_policy.json", "baseline.json", "model_policy.json", "model.enc"):
        require(not (CI / name).exists(), f"{name} already exists; refusing to overwrite reference")
    model = load_model()  # checks the original Lab 2 model SHA before decrypt/encrypt/export
    original = read_json(ROOT / "metrics" / "baseline.json")
    require(file_sha256(ROOT / "models" / "mnist_linear.npz") == original["model_sha256"],
            "Current model is not the baseline model from Lab 2")
    prepare_data()
    with np.load(SAMPLE, allow_pickle=False) as archive:
        x, y = archive["images"], archive["labels"]
    policy = {"version": VERSION, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA,
              "selection": "First 100 official test examples per class, class order 0..9",
              "image_shape": [28, 28], "pixel_dtype": "uint8", "label_dtype": "uint8",
              "min_objects": 1000, "max_objects": 1000, "max_file_bytes": 2 * 1024 * 1024,
              "min_class_fraction": 0.09, "max_class_fraction": 0.11,
              "images_sha256": array_digest(x), "labels_sha256": array_digest(y)}
    data_gate(SAMPLE, policy)
    identity = policy["images_sha256"] + ":" + policy["labels_sha256"]
    baseline = {"metric": "f1_macro", "value": evaluate(model, x, y),
                "dataset_version": VERSION, "data_identity": identity, "sample_size": len(y),
                "model_sha256": original["model_sha256"], "model_version": "lab2-v1",
                "origin": "Original Lab 2 model evaluated once on the fixed CI subset"}
    from cryptography.fernet import Fernet
    key_path = Path.home() / ".config" / "ml-lab3" / "model.key"
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not key_path.exists(), "Secret key already exists; do not overwrite it")
    key = Fernet.generate_key()
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(key)
    (CI / "model.enc").write_bytes(Fernet(key).encrypt((ROOT / "models" / "mnist_linear.npz").read_bytes()))
    write_json(CI / "data_policy.json", policy)
    write_json(CI / "baseline.json", baseline)
    write_json(CI / "model_policy.json", {"allowed_relative_drop": 0.05,
               "dataset_version": VERSION, "data_identity": identity, "sample_size": len(y)})
    print("Initialized CI reference. Secret key stored outside Git:", key_path)
    print("CI macro-F1:", baseline["value"])


def decrypt():
    from cryptography.fernet import Fernet
    key = os.environ.get("MODEL_KEY", "")
    require(bool(key), "Missing MODEL_KEY secret")
    payload = Fernet(key.encode()).decrypt((CI / "model.enc").read_bytes())
    expected = (ROOT / "metrics" / "model.sha256").read_text().split()[0]
    require(hashlib.sha256(payload).hexdigest() == expected, "Decrypted model SHA mismatch")
    path = ROOT / "models" / "mnist_linear.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    load_model()


def run(command):
    REPORTS.mkdir(parents=True, exist_ok=True)
    if command == "init":
        initialize()
    elif command == "decrypt":
        decrypt()
    elif command == "prepare":
        prepare_data()
    elif command == "data":
        path = SAMPLE
        if scenario() == "data":
            # Only a copy is changed. Original MNIST and CI reference remain intact.
            path = ROOT / "data" / "ci_bad_copy.npz"
            with np.load(SAMPLE, allow_pickle=False) as archive:
                np.savez_compressed(path, images=archive["images"][:, :, :27],
                                    labels=archive["labels"], version=archive["version"])
        _, _, report = data_gate(path, read_json(CI / "data_policy.json"))
        write_json(REPORTS / "data.json", report)
    elif command == "evaluate":
        x, y, _ = data_gate(SAMPLE, read_json(CI / "data_policy.json"))
        model = load_model()
        if scenario() == "model":
            # Real constant predictor evaluated on the same sample, not fake metric JSON.
            class ConstantModel:
                def predict_features(self, features):
                    return np.zeros(len(features), dtype=np.uint8)
            model = ConstantModel()
        baseline = read_json(CI / "baseline.json")
        result = {key: baseline[key] for key in
                  ("metric", "dataset_version", "data_identity", "sample_size")}
        result.update(value=evaluate(model, x, y), scenario=scenario())
        write_json(REPORTS / "candidate.json", result)
        report = model_gate(result, baseline, read_json(CI / "model_policy.json"))
        write_json(REPORTS / "model_gate.json", report)
        require(report["passed"], "Model quality below threshold")
    elif command == "security":
        report = read_json(REPORTS / "trivy-image.json")
        result = security_gate(report)
        write_json(REPORTS / "security.json", result)
        require(result["passed"], "CRITICAL image vulnerabilities")
        if scenario() == "security":
            simulated = {"SchemaVersion": 2, "Metadata": {"simulation": True},
                         "Results": [{"Target": "SIMULATED-POLICY-TEST", "Vulnerabilities": [
                             {"VulnerabilityID": "SIMULATED-CRITICAL", "PkgName": "dummy",
                              "InstalledVersion": "0", "FixedVersion": "1", "Severity": "CRITICAL"}]}]}
            write_json(REPORTS / "SIMULATED-security-input.json", simulated)
            result = security_gate(simulated)
            write_json(REPORTS / "SIMULATED-security-result.json", result)
            require(result["passed"], "SIMULATION: security policy rejected CRITICAL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "decrypt", "prepare", "data", "evaluate", "security"])
    args = parser.parse_args()
    try:
        run(args.command)
    except Exception as exc:
        # Do not print secrets or input payloads; errors are saved and failure is propagated.
        write_json(REPORTS / f"{args.command}-status.json",
                   {"exit_code": 1, "error_type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(f"{args.command}: FAILED ({type(exc).__name__}): {exc}") from exc
    write_json(REPORTS / f"{args.command}-status.json", {"exit_code": 0})
    print(args.command + ": OK")


if __name__ == "__main__":
    main()
