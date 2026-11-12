"""Offline container smoke check using the same validated inference function."""
import json
import os

import numpy as np

from src.model import load_model, predict


def main():
    uid = os.getuid()
    if uid == 0:
        raise RuntimeError("Container must run as non-root")
    result = predict(load_model(), np.zeros((28, 28), dtype=np.uint8))
    if result.shape != (1,) or int(result[0]) not in range(10):
        raise RuntimeError("Invalid prediction")
    print(json.dumps({"uid": uid, "prediction": int(result[0]), "status": "ok"}))


if __name__ == "__main__":
    main()
