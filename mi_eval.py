import argparse
import logging
from tqdm import tqdm
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from pytorch_lightning import Trainer

from mine.models.mine import MutualInformationEstimator
from minepy import MINE

from dmz.datasets import ImageDataset
from ddpm_train import create_gaussian_diffusion

logging.getLogger().setLevel(logging.ERROR)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_train", type=str)
    parser.add_argument("--latents_test", type=str)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--data_dir_train", type=str, default="")
    parser.add_argument("--data_dir_test", type=str, default="")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--images_train", type=str, default="")
    parser.add_argument("--images_test", type=str, default="")
    parser.add_argument("--binary", action="store_true", default=False)
    args = parser.parse_args()
    return args


class DiffusionDataset(Dataset):
    def __init__(self, dataset_img, dataset_z, diffusion, t, binary):
        super().__init__()
        self.dataset_img = dataset_img
        self.dataset_z = dataset_z
        self.diffusion = diffusion
        self.t = t
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgs = []
        self.binary = binary
        for img, _ in self.dataset_img:
            img = torch.from_numpy(img).to(self.device).unsqueeze(0)
            if self.t == 1:
                img = torch.randn_like(img).to(self.device)
            else:
                t = int(self.diffusion.num_timesteps * self.t)
                t_tensor = torch.tensor([t] * img.shape[0], device=self.device)
                img = self.diffusion.q_sample(img, t_tensor)
            imgs.append(img)
        self.imgs = imgs

    def __len__(self):
        return len(self.dataset_z)

    def __getitem__(self, idx):
        img = self.imgs[idx]

        img = img.flatten()
        z = self.dataset_z[idx][0].to(self.device)

        if self.binary:
            z = (z > 0.5).float()

        return img, z


class CustomDataset(Dataset):
    def __init__(self, img_path, dataset_z, t, binary):
        super().__init__()
        self.dataset_z = dataset_z
        self.t = t
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.binary = binary
        if "_0_" in img_path:
            self.dataset_img = []
            self.paths = []
            for i in range(20):
                path = img_path.replace("_0_", f"_{i}_")
                if os.path.exists(path):
                    print(f"Loading {path}")
                    self.dataset_img.append(np.load(path)["arr_0"][t])
            self.dataset_img = self.dataset_img
        else:
            self.paths = []
            self.dataset_img = [np.load(img_path)["arr_0"][t]]
        self.count = torch.ones(len(self.dataset_img[0])).long()

    def __len__(self):
        return len(self.dataset_img[0])

    def __getitem__(self, idx):
        if len(self.paths) == 0:
            r = self.count[idx]
            self.count[idx] += 1
            self.count[idx] = (self.count[idx] % len(self.dataset_img))
            img = self.dataset_img[r][idx]
        else:
            r = random.randint(0, len(self.paths)-1)
            img = np.load(self.paths[r], mmap_mode="r")["arr_0"][self.t]

        img = 2 * (torch.from_numpy(img).to(self.device).float() / 255.0) - 1
        img = img.flatten()
        z = self.dataset_z[idx][0]

        if self.binary:
            z = (z > 0.5).float()
        return img, z


def main():
    args = args_parser()
    print(args)
    batch_size = args.batch_size
    epochs = 10
    lr = 1e-4

    train_loaded = torch.load(args.latents_train, weights_only=False)
    test_loaded = torch.load(args.latents_test, weights_only=False)
    X_train, y_train = train_loaded["latents"].float(), train_loaded["classes"]
    X_test, y_test = test_loaded["latents"].float(), test_loaded["classes"]

    latents_train = TensorDataset(X_train, y_train)
    latents_test = TensorDataset(X_test, y_test)
    diffusion_args = {
        "diffusion_steps": 1000,
        "variance_type": "learned_range",
        "noise_schedule": "cosine",
        "timestep_respacing": "",
        "rescale_timesteps": False,
        "rescale_learned_sigmas": True,
        "input_pertub": 0.0,
    }
    diffusion = create_gaussian_diffusion(**diffusion_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.images_train != "":
        ts = range(20)
    else:
        ts = [0.1 * float(t) for t in range(10)] + [(0.9 + 0.01 * float(t)) for t in range(1, 11)]
        ts = ts[::-1]
    print(ts)

    for t in ts:
        print("t=", t)
        if args.images_train != "":
            dataset_train = CustomDataset(args.images_train, latents_train, t, binary=args.binary)
            dataset_test = CustomDataset(args.images_test, latents_test, t, binary=args.binary)
        else:
            img_train = ImageDataset(
                args.image_size,
                args.data_dir_train,
                classes=True,
                random_crop=False,
                random_flip=False,
            )
            img_test = ImageDataset(
                args.image_size,
                args.data_dir_test,
                classes=True,
                random_crop=False,
                random_flip=False,
            )
            dataset_train = DiffusionDataset(img_train, latents_train, diffusion, t, binary=args.binary)
            dataset_test = DiffusionDataset(img_test, latents_test, diffusion, t, binary=args.binary)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        loss = "mine"
        kwargs = {
            "lr": lr,
            "batch_size": batch_size,
            "train_loader": dataloader_train,
            "test_loader": dataloader_test,
            "alpha": 1.0,
        }

        model = MutualInformationEstimator(
            args.image_size * args.image_size * 3,
            X_train.shape[-1],
            loss=loss,
            **kwargs,
        ).to(device)

        trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
        trainer.fit(model)
        with torch.no_grad():
            trainer.test(ckpt_path="best")
            print(f"t={t} MINE {model.avg_test_mi}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
