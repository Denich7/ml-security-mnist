import hashlib
import io
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist_linear.npz"
MODEL_HASH_PATH = ROOT / "metrics" / "model.sha256"

MAX_BATCH = 64
MAX_FILE_BYTES = 64 * 1024
MAX_PIXELS = 28 * 28


class InputError(ValueError):
    """Контролируемая ошибка входных данных."""


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def preprocess(value, max_batch=MAX_BATCH):
    if not isinstance(value, np.ndarray):
        raise InputError("Ожидается массив NumPy.")

    if value.size == 0:
        raise InputError("Пустой вход запрещён.")

    if value.ndim == 2:
        if value.shape != (28, 28):
            raise InputError("Размер изображения должен быть 28 × 28.")
        batch = value[np.newaxis, ...]

    elif value.ndim == 3:
        if value.shape[1:] != (28, 28):
            raise InputError("Ожидается форма (N, 28, 28).")
        batch = value

    else:
        raise InputError("Ожидается изображение или пакет изображений.")

    if not 1 <= len(batch) <= max_batch:
        raise InputError(f"Размер пакета должен быть от 1 до {max_batch}.")

    if batch.dtype.kind not in "uif":
        raise InputError("Пиксели должны быть действительными числами.")

    if not np.isfinite(batch).all():
        raise InputError("NaN и Inf запрещены.")

    if np.any(batch < 0) or np.any(batch > 255):
        raise InputError("Значения пикселей должны находиться в 0–255.")

    # Фиксированное преобразование: оно не обучается на данных.
    return batch.reshape(len(batch), 784).astype(np.float32) / 255.0


class LinearModel:
    def __init__(self, weights, bias, classes):
        self.weights = weights
        self.bias = bias
        self.classes = classes

    def predict_features(self, features):
        scores = features @ self.weights.T + self.bias
        return self.classes[np.argmax(scores, axis=1)]


def predict(model, value, max_batch=MAX_BATCH):
    features = preprocess(value, max_batch=max_batch)
    return model.predict_features(features)


def predict_png(model, content, max_bytes=MAX_FILE_BYTES):
    if not isinstance(content, bytes):
        raise InputError("Файл должен быть передан как bytes.")

    if not content:
        raise InputError("Пустой файл запрещён.")

    if len(content) > max_bytes:
        raise InputError("Превышен допустимый размер файла.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(content)) as image:
                if image.format != "PNG":
                    raise InputError("Поддерживается только фактический формат PNG.")

                width, height = image.size

                # Проверяем заголовок до полной загрузки пикселей.
                if width * height > MAX_PIXELS:
                    raise InputError("Превышен лимит числа пикселей.")

                if image.size != (28, 28):
                    raise InputError("Размер PNG должен быть 28 × 28.")

                if image.mode != "L":
                    raise InputError("PNG должен быть в оттенках серого, режим L.")

                image.load()
                pixels = np.array(image, copy=True)

    except InputError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise InputError("Файл повреждён или не может быть декодирован.") from exc

    return predict(model, pixels)


def save_model(classifier):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        MODEL_PATH,
        weights=classifier.coef_,
        bias=classifier.intercept_,
        classes=classifier.classes_,
    )

    digest = file_sha256(MODEL_PATH)
    MODEL_HASH_PATH.write_text(
        f"{digest}  models/mnist_linear.npz\n",
        encoding="utf-8",
    )

    return digest


def load_model():
    expected = MODEL_HASH_PATH.read_text(encoding="utf-8").split()[0]

    if file_sha256(MODEL_PATH) != expected:
        raise ValueError("SHA256 модели не совпадает с эталоном.")

    with np.load(MODEL_PATH, allow_pickle=False) as archive:
        weights = archive["weights"].copy()
        bias = archive["bias"].copy()
        classes = archive["classes"].copy()

    if weights.shape != (10, 784) or bias.shape != (10,):
        raise ValueError("Некорректная форма параметров модели.")

    if not np.array_equal(classes, np.arange(10)):
        raise ValueError("Некорректный список классов.")

    if weights.dtype.kind not in "fiu" or bias.dtype.kind not in "fiu":
        raise ValueError("Некорректный тип параметров.")

    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("Некорректные значения параметров.")

    return LinearModel(weights, bias, classes)

