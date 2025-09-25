import os

import cv2
import tempfile
from tqdm.auto import tqdm

import numpy as np
import torchvision


DATA_DIR = "data"
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    data_dir = f"{DATA_DIR}/cifar10"
    os.makedirs(data_dir, exist_ok=True)
    idx = 0
    for split in ["train", "test"]:
        out_dir = f"{data_dir}/cifar10_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(root=tmp_dir, train=split == "train", download=True)

        print("saving images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            idx = idx + 1
            filename = f"{out_dir}/{CLASSES[label]}_{idx:05d}.png"
            image.save(filename)


if __name__ == "__main__":
    main()
