"""
Dataset handler from [guided-diffusion](https://github.com/openai/guided-diffusion) and
[DDPM-IP](https://github.com/forever208/DDPM-IP) adjusted to provide labels.
"""

import blobfile as bf
from mpi4py import MPI
import pandas as pd
from PIL import Image
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader

from guided_diffusion.image_datasets import (
    _list_image_files_recursively,
    random_crop_arr,
    center_crop_arr,
)

DATA_DIR = f"data"
CELEBA_ATTRS_PATH = f"{DATA_DIR}/celeba/list_attr_celeba.txt"
CELEBA_MAPPING_PATH = f"{DATA_DIR}/celebahq/celebahq_celeba_mapping.txt"

CIFAR_CLASSES = [
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
]
CELEBA_index_to_label = {
    0: "5_o_Clock_Shadow",
    1: "Arched_Eyebrows",
    2: "Attractive",
    3: "Bags_Under_Eyes",
    4: "Bald",
    5: "Bangs",
    6: "Big_Lips",
    7: "Big_Nose",
    8: "Black_Hair",
    9: "Blond_Hair",
    10: "Blurry",
    11: "Brown_Hair",
    12: "Bushy_Eyebrows",
    13: "Chubby",
    14: "Double_Chin",
    15: "Eyeglasses",
    16: "Goatee",
    17: "Gray_Hair",
    18: "Heavy_Makeup",
    19: "High_Cheekbones",
    20: "Male",
    21: "Mouth_Slightly_Open",
    22: "Mustache",
    23: "Narrow_Eyes",
    24: "No_Beard",
    25: "Oval_Face",
    26: "Pale_Skin",
    27: "Pointy_Nose",
    28: "Receding_Hairline",
    29: "Rosy_Cheeks",
    30: "Sideburns",
    31: "Smiling",
    32: "Straight_Hair",
    33: "Wavy_Hair",
    34: "Wearing_Earrings",
    35: "Wearing_Hat",
    36: "Wearing_Lipstick",
    37: "Wearing_Necklace",
    38: "Wearing_Necktie",
    39: "Young",
}




def get_celeba_attrs(path, attrs, shift):
    attr_idx = int(path.split("/")[-1].split(".")[0])
    return attrs[attr_idx + shift]


def get_cifar10_class(path):
    cls = path.split("/")[-1].split("_")[0]
    return np.array(CIFAR_CLASSES.index(cls), dtype=np.int64)


def get_celebahq_mapping():
    mapping = dict()
    with open(CELEBA_MAPPING_PATH) as f:
        for l in f.readlines():
            a, b = l.split(" ")
            a, b = int(a), int(b.replace(".jpg", "")) - 1
            mapping[a] = b
    return mapping


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    @param data_dir: a dataset directory.
    @param batch_size: the batch size of each returned pair.
    @param image_size: the size to which images are resized.
    @param class_cond: if True, include a "y" key in returned dicts for class label.
                       If classes are not available and this is true, an exception will be raised.
    @param deterministic: if True, yield results in a deterministic order.
    @param random_crop: if True, randomly crop the images for augmentation.
    @param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # partition the whole dataset into each sub-dataset based on the num_of_GPUs
    dataset = ImageDataset(
        image_size,
        data_dir,
        classes=class_cond,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_dir,
        classes=False,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        image_paths = _list_image_files_recursively(data_dir)
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

        # Determine dataset
        self.is_grey = False
        if "celebahq" in data_dir.lower():
            self.dataset_name = "celebahq"
        elif "celeba" in data_dir.lower():
            self.dataset_name = "celeba"
        elif "cifar10" in data_dir.lower():
            self.dataset_name = "cifar10"
        elif "edges64" in data_dir.lower():
            self.dataset_name = "edges"
            self.is_grey = True
        elif "handbags64" in data_dir.lower():
            self.dataset_name = "handbags"
        else:
            assert False

        self.local_labels = None

        if classes:
            # Get correct class labels, assuming that provided data dir contains images from train, val, test or val+test.
            celeba_attrs = (pd.read_csv(CELEBA_ATTRS_PATH, skiprows=1, sep="\s+").to_numpy() > 0).astype(int)
            if self.dataset_name == "celeba":
                shift = {
                    162770: 0,
                    19867: 162770,
                    19962: 162770 + 19867,
                    19867 + 19962: 162770,
                }[len(self.local_images)]
                self.local_labels = [get_celeba_attrs(path, celeba_attrs, shift) for path in self.local_images]
            elif self.dataset_name == "celebahq":
                mapping = get_celebahq_mapping()
                shift = {24183: 0, 5817: 24183}[len(self.local_images)]
                self.local_labels = list()
                for path in self.local_images:
                    idx = int(path.split("/")[-1].split(".")[0])
                    self.local_labels.append(celeba_attrs[mapping[idx + shift]])
            elif self.dataset_name == "cifar10":
                self.local_labels = [get_cifar10_class(path) for path in self.local_images]
            else:
                self.local_labels = None

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("L") if self.is_grey else pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]

        out_dict = {}
        if self.local_labels is not None:
            out_dict["y"] = self.local_labels[idx]
        return np.transpose(arr, [2, 0, 1]), out_dict
