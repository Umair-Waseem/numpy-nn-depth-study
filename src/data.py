import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}
PIXEL_MAX = 255.0
# Anchored to the repository root so the cache is found from any directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download(data_dir=DATA_DIR):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for filename in FILES.values():
        destination = directory / filename
        if destination.exists():
            continue
        print(f"Downloading {filename}")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)

    return directory


def read_idx(path):
    with gzip.open(path, "rb") as handle:
        # The header holds a magic number then one 32-bit size per dimension.
        _, _, _, dimensions = struct.unpack(">BBBB", handle.read(4))
        shape = struct.unpack(f">{dimensions}I", handle.read(4 * dimensions))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)


def load_binary_pair(class_a=0, class_b=6, data_dir=DATA_DIR):
    directory = download(data_dir)

    def prepare(images_file, labels_file):
        images = read_idx(directory / images_file)
        labels = read_idx(directory / labels_file)
        selected = (labels == class_a) | (labels == class_b)
        # Transposing puts one example per column, giving X shape (features, m).
        X = images[selected].reshape(selected.sum(), -1).T / PIXEL_MAX
        Y = (labels[selected] == class_b).astype(float).reshape(1, -1)
        return X, Y

    X_train, Y_train = prepare(FILES["train_images"], FILES["train_labels"])
    X_test, Y_test = prepare(FILES["test_images"], FILES["test_labels"])

    return X_train, Y_train, X_test, Y_test
