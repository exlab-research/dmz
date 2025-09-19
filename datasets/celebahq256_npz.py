import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np

DATA_DIR = "data"
N = 30000


def save_mapping(data_dir):
    mapping = []
    in_test = []

    with open(f"{data_dir}/CelebA-HQ-to-CelebA-mapping.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("idx"):
                continue
            filename = line.split(" ")[-1]
            idx = int(filename.replace(".jpg", ""))
            if idx > 162770:
                in_test.append(True)
            else:
                in_test.append(False)
            mapping.append(filename)

    arranged_mapping = []
    for i, test in enumerate(in_test):
        if not test:
            arranged_mapping.append(mapping[i])
    for i, test in enumerate(in_test):
        if test:
            arranged_mapping.append(mapping[i])

    assert len(arranged_mapping) == N
    assert len(in_test) == N
    output_path = f"{data_dir}/celebahq_celeba_mapping.txt"
    with open(output_path, "w") as f:
        for i, map in enumerate(arranged_mapping):
            f.write(f"{i} {map}")
    print(f"CelebA-HQ to CelebA mapping saved to {output_path}")
    return in_test


def imgs_to_npz():
    data_dir = f"{DATA_DIR}/celebahq"
    in_test = save_mapping(data_dir)
    npz_train, npz_test = [], []
    for i, test in tqdm(enumerate(in_test), total=N):
        img_arr = cv2.imread(f"{data_dir}/CelebA-HQ-img/{i}.jpg")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_arr, (256, 256))
        if test:
            npz_test.append(resized_img)
        else:
            npz_train.append(resized_img)

    output_npz = np.array(npz_train)
    output_path = f"{data_dir}/celebahq256_train.npz"
    np.savez(output_path, output_npz)
    print(f"{output_npz.shape} size array saved into {output_path}")

    output_npz = np.array(npz_test)
    output_path = f"{data_dir}/celebahq256_test.npz"
    np.savez(output_path, output_npz)
    print(f"{output_npz.shape} size array saved into {output_path}")


if __name__ == "__main__":
    imgs_to_npz()
