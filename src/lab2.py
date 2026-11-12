import argparse
import hashlib
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.model import (
    MAX_BATCH,
    ROOT,
    file_sha256,
    load_model,
    predict,
    preprocess,
    save_model,
)


SEED = 42
DATA_PATH = ROOT / "data" / "mnist.npz"
REPORTS = ROOT / "reports" / "lab2"
METRICS = ROOT / "metrics"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def image_hashes(images):
    return [
        hashlib.sha256(image.tobytes()).hexdigest()
        for image in images
    ]


def load_data():
    expected = (
        ROOT / "reports" / "dataset.sha256"
    ).read_text(encoding="utf-8").split()[0]

    actual = file_sha256(DATA_PATH)

    if actual != expected:
        raise ValueError("Датасет не совпадает с эталоном ЛР1.")

    with np.load(DATA_PATH, allow_pickle=False) as archive:
        x_train = archive["x_train"].copy()
        y_train = archive["y_train"].copy()
        x_test = archive["x_test"].copy()
        y_test = archive["y_test"].copy()

    for images, labels, count in (
        (x_train, y_train, 60000),
        (x_test, y_test, 10000),
    ):
        if images.shape != (count, 28, 28):
            raise ValueError("Неожиданная форма изображений.")

        if images.dtype != np.uint8:
            raise ValueError("Ожидается uint8 для пикселей.")

        if labels.shape != (count,) or labels.dtype.kind not in "ui":
            raise ValueError("Некорректная форма или тип меток.")

        if not np.isfinite(images).all():
            raise ValueError("Обнаружены NaN или Inf.")

        if not np.isin(labels, np.arange(10)).all():
            raise ValueError("Обнаружены некорректные метки.")

    return x_train, y_train, x_test, y_test, actual


def prepare_data():
    x, y, x_test, y_test, digest = load_data()

    train_hashes = image_hashes(x)
    test_hashes = image_hashes(x_test)
    test_set = set(test_hashes)

    labels_by_hash = {}

    for key, label in zip(train_hashes, y):
        labels_by_hash.setdefault(key, set()).add(int(label))

    conflicting = {
        key for key, labels in labels_by_hash.items()
        if len(labels) > 1
    }

    seen = set()
    keep = []

    # Сохраняем официальный тест.
    # Из обучающего набора удаляем его точные пересечения с тестом,
    # противоречиво размеченные изображения и повторные копии.
    for index, key in enumerate(train_hashes):
        if key in test_set or key in conflicting or key in seen:
            continue
        seen.add(key)
        keep.append(index)

    keep = np.asarray(keep, dtype=np.int64)

    train_indices, validation_indices = train_test_split(
        keep,
        test_size=0.2,
        random_state=SEED,
        stratify=y[keep],
    )

    train_set = {train_hashes[index] for index in train_indices}
    validation_set = {
        train_hashes[index] for index in validation_indices
    }

    if train_set & validation_set or train_set & test_set:
        raise ValueError("Обнаружено пересечение выборок.")

    if validation_set & test_set:
        raise ValueError("Валидация пересекается с тестом.")

    report = {
        "dataset_sha256": digest,
        "seed": SEED,
        "original_train_size": len(x),
        "original_test_size": len(x_test),
        "pixel_dtype": str(x.dtype),
        "label_dtype": str(y.dtype),
        "pixel_min": int(x.min()),
        "pixel_max": int(x.max()),
        "missing_values": 0,
        "train_duplicate_extra_copies": len(x) - len(set(train_hashes)),
        "test_duplicate_extra_copies": len(x_test) - len(test_set),
        "train_test_shared_unique_images": len(set(train_hashes) & test_set),
        "conflicting_train_image_groups": len(conflicting),
        "removed_from_original_train": int(len(x) - len(keep)),
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(x_test),
        "train_class_counts": np.bincount(
            y[train_indices], minlength=10
        ).tolist(),
        "validation_class_counts": np.bincount(
            y[validation_indices], minlength=10
        ).tolist(),
        "test_class_counts": np.bincount(
            y_test, minlength=10
        ).tolist(),
        "split_policy": (
            "Official test retained. Exact image overlaps removed "
            "from training pool; training duplicates and conflicting "
            "training groups removed before stratified split."
        ),
        "limitations": [
            "Writer identifiers are unavailable in this archive.",
            "Exact deduplication does not detect near-duplicate images.",
            "Official test duplicates are retained for benchmark comparability.",
            "MNIST does not represent arbitrary photos or document layouts.",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "data_check.json", report)

    np.savez_compressed(
        ROOT / "data" / "lab2_split.npz",
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    return (
        x[train_indices],
        y[train_indices],
        x[validation_indices],
        y[validation_indices],
        x_test,
        y_test,
        digest,
    )


def features(images):
    # Для обучения применяем ровно ту же предобработку,
    # но разрешаем весь учебный набор.
    return preprocess(images, max_batch=len(images))


def predict_batches(model, images):
    return np.concatenate([
        predict(model, images[start:start + MAX_BATCH])
        for start in range(0, len(images), MAX_BATCH)
    ])


def scores(truth, predicted):
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(
            precision_score(truth, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, predicted, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
    }


def plot_results(images, truth, predicted):
    ConfusionMatrixDisplay.from_predictions(
        truth,
        predicted,
        labels=np.arange(10),
        cmap="Blues",
        colorbar=False,
    )
    plt.title("MNIST: confusion matrix")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close()

    wrong = np.flatnonzero(truth != predicted)[:8]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for axis in axes.flat:
        axis.axis("off")

    examples = []

    for axis, index in zip(axes.flat, wrong):
        axis.imshow(images[index], cmap="gray")
        axis.set_title(
            f"True: {truth[index]}, predicted: {predicted[index]}"
        )
        examples.append({
            "test_index": int(index),
            "true": int(truth[index]),
            "predicted": int(predicted[index]),
        })

    fig.tight_layout()
    fig.savefig(REPORTS / "errors.png", dpi=150)
    plt.close(fig)

    write_json(REPORTS / "error_examples.json", examples)


def robustness(model, images, labels):
    # 20 фиксированных объектов: по два изображения каждого класса.
    indices = np.concatenate([
        np.flatnonzero(labels == label)[:2]
        for label in range(10)
    ])

    original = images[indices]
    truth = labels[indices]

    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 8, size=original.shape)

    changed = np.clip(
        original.astype(np.float32) + noise, 0, 255
    ).astype(np.uint8)

    before = predict(model, original)
    after = predict(model, changed)

    # Нетипичный, но формально допустимый вход:
    # равномерный шум вместо изображения цифры.
    unusual = rng.integers(
        0, 256, size=(28, 28), dtype=np.uint8
    )
    unusual_prediction = int(predict(model, unusual)[0])

    rows = [
        {
            "test_index": int(index),
            "true": int(label),
            "original_prediction": int(first),
            "changed_prediction": int(second),
            "prediction_changed": bool(first != second),
        }
        for index, label, first, second in zip(
            indices, truth, before, after
        )
    ]

    result = {
        "seed": SEED,
        "sample_size": len(indices),
        "change": "Gaussian pixel noise, sigma=8 on the 0-255 scale",
        "changed_answer_fraction": float(np.mean(before != after)),
        "accuracy_before": float(accuracy_score(truth, before)),
        "accuracy_after": float(accuracy_score(truth, after)),
        "assumption": "Small noise is assumed to preserve digit identity.",
        "unusual_input": "Uniform random pixel noise, not a handwritten digit.",
        "unusual_prediction": unusual_prediction,
        "unusual_input_rejected": False,
        "conclusion": (
            "Schema validation does not detect semantic anomalies. "
            "The classifier always selects one of ten classes."
        ),
        "next_measure": (
            "Evaluate noise augmentation and a rejection policy on "
            "a separate validation set of non-digit images."
        ),
        "objects": rows,
    }

    write_json(REPORTS / "robustness.json", result)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for column in range(5):
        axes[0, column].imshow(original[column], cmap="gray")
        axes[0, column].set_title(f"Original: {before[column]}")
        axes[1, column].imshow(changed[column], cmap="gray")
        axes[1, column].set_title(f"Noise: {after[column]}")

    for axis in axes.flat:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(REPORTS / "robustness.png", dpi=150)
    plt.close(fig)

    plt.imshow(unusual, cmap="gray")
    plt.title(f"Random noise; predicted class: {unusual_prediction}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(REPORTS / "unusual_input.png", dpi=150)
    plt.close()


def train_run():
    baseline_path = METRICS / "baseline.json"

    if baseline_path.exists():
        raise FileExistsError(
            "baseline.json уже существует. "
            "Для повторной оценки используйте команду evaluate."
        )

    x_train, y_train, x_val, y_val, x_test, y_test, digest = prepare_data()

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=50,
        tol=0.001,
        random_state=SEED,
        average=True,
        n_jobs=1,
    )

    start = time.perf_counter()
    classifier.fit(features(x_train), y_train)
    elapsed = time.perf_counter() - start

    validation_prediction = classifier.predict(features(x_val))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_prediction = dummy.predict(np.zeros((len(y_test), 1)))

    model_digest = save_model(classifier)
    model = load_model()

    # Проверка, что экспорт параметров сохраняет предсказания sklearn.
    exported_prediction = predict(model, x_test[:64])
    original_prediction = classifier.predict(features(x_test[:64]))

    if not np.array_equal(exported_prediction, original_prediction):
        raise ValueError("Экспорт модели изменил предсказания.")

    test_prediction = predict_batches(model, x_test)
    test_scores = scores(y_test, test_prediction)

    baseline = {
        "model_version": "lab2-v1",
        "model_type": "SGDClassifier linear logistic classifier",
        "primary_metric": "f1_macro",
        "primary_value": test_scores["f1_macro"],
        "test_size": len(y_test),
        "dataset_version": f"mnist-npz-sha256:{digest}",
        "model_sha256": model_digest,
        "test_metrics": test_scores,
        "validation_metrics": scores(y_val, validation_prediction),
        "dummy_test_metrics": scores(y_test, dummy_prediction),
        "per_class": classification_report(
            y_test,
            test_prediction,
            output_dict=True,
            zero_division=0,
        ),
    }

    parameters = {
        "seed": SEED,
        "model_parameters": classifier.get_params(),
        "training_seconds": elapsed,
        "iterations": int(classifier.n_iter_),
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "test_size": len(y_test),
        "preprocessing": "reshape to 784, float32, divide by 255",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "selection_policy": (
            "Parameters fixed before test evaluation; "
            "no hyperparameter search on test data."
        ),
    }

    write_json(METRICS / "training.json", parameters)
    plot_results(x_test, y_test, test_prediction)
    robustness(model, x_test, y_test)

    # Baseline записывается только при обучении новой базовой модели.
    write_json(baseline_path, baseline)

    print(json.dumps(baseline, ensure_ascii=False, indent=2))


def evaluate_run():
    _, _, x_test, y_test, digest = load_data()
    model = load_model()
    predicted = predict_batches(model, x_test)

    result = {
        "dataset_version": f"mnist-npz-sha256:{digest}",
        "test_size": len(y_test),
        "metrics": scores(y_test, predicted),
    }

    write_json(METRICS / "evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def predict_example():
    _, _, x_test, y_test, _ = load_data()
    model = load_model()

    print({
        "test_index": 0,
        "true": int(y_test[0]),
        "prediction": int(predict(model, x_test[0])[0]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "predict"])
    args = parser.parse_args()

    if args.command == "train":
        train_run()
    elif args.command == "evaluate":
        evaluate_run()
    else:
        predict_example()


if __name__ == "__main__":
    main()
