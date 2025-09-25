import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np


DATA_DIR = "data"


def imgs_to_npz(split):
    npz_sketch = []
    npz_photo = []
    data_dir = f"{DATA_DIR}/edges2handbags"
    files = os.listdir(f"{data_dir}/{split}")
    files = sorted(files)
    for img in tqdm(files, total=len(files)):
        img = Image.open(f"{data_dir}/{split}/{img}")

        width, height = img.size
        left_image = img.crop((0, 0, width // 2, height))
        right_image = img.crop((width // 2, 0, width, height))
        left_image = left_image.resize((64, 64))
        right_image = right_image.resize((64, 64))

        left_image_greyscale = left_image.convert("L")
        left_image_array = np.array(left_image_greyscale)[:, :, np.newaxis]

        right_image_array = np.array(right_image)
        npz_sketch.append(left_image_array)
        npz_photo.append(right_image_array)

    sketch_npz = np.array(npz_sketch)
    npz_photo = np.array(npz_photo)
    path = f"{data_dir}/edges64_{split}.npz"
    np.savez(path, sketch_npz)
    print(f"{sketch_npz.shape} size array saved into {path}")
    path = f"{data_dir}/handbags64_{split}.npz"
    np.savez(path, npz_photo)
    print(f"{npz_photo.shape} size array saved into {path}")


if __name__ == "__main__":
    for split in ["train", "test"]:
        imgs_to_npz(split)
