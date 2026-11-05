import io
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from src.model import InputError, load_model, predict, predict_png


def png_bytes(array, file_format="PNG"):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=file_format)
    return stream.getvalue()


@pytest.mark.parametrize(
    "value",
    [
        "not an image",
        np.zeros((28, 27), dtype=np.uint8),
        np.empty((0, 28, 28), dtype=np.uint8),
        np.full((28, 28), np.nan),
        np.full((28, 28), np.inf),
        np.full((28, 28), -1.0),
        np.full((28, 28), 256.0),
        np.full((28, 28), "x"),
    ],
    ids=[
        "wrong_type",
        "wrong_shape",
        "empty_batch",
        "nan",
        "inf",
        "negative_pixel",
        "pixel_above_255",
        "string_pixels",
    ],
)
def test_invalid_array_does_not_call_model(value):
    model = Mock()

    with pytest.raises(InputError):
        predict(model, value)

    model.predict_features.assert_not_called()


def test_batch_limit_does_not_call_model():
    model = Mock()
    value = np.zeros((3, 28, 28), dtype=np.uint8)

    with pytest.raises(InputError):
        predict(model, value, max_batch=2)

    model.predict_features.assert_not_called()


def test_corrupted_file_does_not_call_model():
    model = Mock()

    with pytest.raises(InputError):
        predict_png(model, b"not a real image")

    model.predict_features.assert_not_called()


def test_file_size_limit_does_not_call_model():
    model = Mock()

    with pytest.raises(InputError):
        predict_png(model, b"x" * 33, max_bytes=32)

    model.predict_features.assert_not_called()


def test_pixel_limit_does_not_call_model():
    model = Mock()
    content = png_bytes(np.zeros((29, 28), dtype=np.uint8))

    with pytest.raises(InputError):
        predict_png(model, content)

    model.predict_features.assert_not_called()


def test_unsupported_format_does_not_call_model():
    model = Mock()
    content = png_bytes(
        np.zeros((28, 28), dtype=np.uint8),
        file_format="JPEG",
    )

    with pytest.raises(InputError):
        predict_png(model, content)

    model.predict_features.assert_not_called()


def test_rgb_png_does_not_call_model():
    model = Mock()
    content = png_bytes(np.zeros((28, 28, 3), dtype=np.uint8))

    with pytest.raises(InputError):
        predict_png(model, content)

    model.predict_features.assert_not_called()


def test_correct_array_calls_model_with_scaled_features():
    model = Mock()
    model.predict_features.return_value = np.array([7])

    value = np.full((28, 28), 255, dtype=np.uint8)
    result = predict(model, value)

    assert result.tolist() == [7]
    model.predict_features.assert_called_once()

    features = model.predict_features.call_args.args[0]
    assert features.shape == (1, 784)
    np.testing.assert_array_equal(features, np.ones((1, 784)))


def test_correct_png_calls_model():
    model = Mock()
    model.predict_features.return_value = np.array([0])

    content = png_bytes(np.zeros((28, 28), dtype=np.uint8))
    result = predict_png(model, content)

    assert result.tolist() == [0]
    model.predict_features.assert_called_once()


def test_saved_model_can_predict():
    model = load_model()
    result = predict(model, np.zeros((28, 28), dtype=np.uint8))

    assert result.shape == (1,)
    assert int(result[0]) in range(10)

