import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np

DATA_DIR = "data"


def imgs_to_npz():
    data_dir = f"{DATA_DIR}/celeba"
    npz = []
    files = os.listdir(f"{data_dir}/img_align_celeba")
    files = sorted(files)
    for img in tqdm(files, total=len(files)):
        img_arr = cv2.imread(f"{data_dir}/img_align_celeba/{img}")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_arr, (64, 64))
        npz.append(resized_img)

    output_npz = np.array(npz[:162770])
    output_path = f"{data_dir}/celeba64_train.npz"
    np.savez(output_path, output_npz)
    print(f"{output_npz.shape} size array saved into {output_path}")

    output_npz = np.array(npz[162770:])
    output_path = f"{data_dir}/celeba64_val_test.npz"
    np.savez(output_path, output_npz)
    print(f"{output_npz.shape} size array saved into {output_path}")


if __name__ == "__main__":
    imgs_to_npz()
