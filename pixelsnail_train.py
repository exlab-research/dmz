import argparse
import os
import sys
from tqdm import tqdm
import warnings

import numpy as np
import torch
from torch import amp, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets

from dmz.pixelsnail import PixelSNAIL

warnings.filterwarnings("ignore", category=FutureWarning)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_train", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--n_bl", type=int, default=2)
    parser.add_argument("--n_res_bl", type=int, default=2)
    parser.add_argument("--res_ch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()
    return args


def cross_entropy_with_eps(logits, targets, eps=1e-8):
    """
    Computes the cross-entropy loss with numerical stability improvements.

    Args:
        logits: Tensor of shape (batch_size, num_classes), raw model outputs.
        targets: Tensor of shape (batch_size,), containing class indices (not one-hot).
        eps: Small value to prevent log(0).

    Returns:
        Scalar loss value.
    """
    logits = logits.permute(0, 2, 3, 1)
    logits = logits.view(-1, 2)
    targets = targets.view(-1)

    probs = F.softmax(logits, dim=-1) + eps
    log_probs = torch.log(probs)
    loss = -log_probs[torch.arange(logits.shape[0]), targets]

    return loss.mean()


def train(epoch, loader, model, optimizer, device, scaler):
    loader = tqdm(loader)
    model.train()
    criterion = cross_entropy_with_eps
    accs = list()

    for i, img in enumerate(loader):
        img = img[0].to(device)
        model.zero_grad()
        with amp.autocast("cuda"):
            out, _ = model(img)
            loss = criterion(out, img)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = out.argmax(1)
        correct = (pred == img).float()
        accuracy = correct.sum() / img.numel()
        accs.append(accuracy.cpu().item())
        loader.set_description(f"TRAIN | Epoch: {epoch} | Loss: {loss.item():.4f} | Acc: {np.mean(accs):.4f}")
    return np.mean(accs)


def eval(epoch, loader, model, device):
    loader = tqdm(loader)
    model.eval()
    criterion = cross_entropy_with_eps
    accs = list()
    with torch.no_grad():
        for i, img in enumerate(loader):
            img = img[0].to(device)
            out, _ = model(img)
            loss = criterion(out, img)
            _, pred = out.max(1)
            correct = (pred == img).float()
            accuracy = correct.sum() / img.numel()
            accs.append(accuracy.cpu().item())
            loader.set_description(f"EVAL | Epoch: {epoch} | Loss: {loss.item():.4f} | Acc: {np.mean(accs):.4f}")
    print(f"EVAL epoch {epoch}: {np.mean(accs):.5f}")
    return np.mean(accs)


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == "__main__":
    args = args_parser()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    train_loaded = torch.load(args.latents_train, map_location="cpu")
    X = train_loaded["latents"]
    if args.binary:
        X = (X > 0.5).long()
        X = X.reshape(X.shape[0], -1, 2)[:, :, 1]
    X = X.reshape(X.shape[0], -1, args.dim)

    perm = np.random.RandomState(seed=42).permutation(len(X))
    N = int(0.8 * len(X))
    M = int(0.6 * len(X))
    X = X[perm]
    X_train, X_val, X_test = X, X[M:N], X[N:]

    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    test_dataset = TensorDataset(X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_args = {
        "shape": X[0].shape,
        "n_class": X.max() + 1,
        "channel": args.ch,
        "kernel_size": args.kernel,
        "n_block": args.n_bl,
        "n_res_block": args.n_res_bl,
        "res_channel": args.res_ch,
    }
    model = PixelSNAIL(**model_args)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = amp.GradScaler()

    dir_path = os.path.dirname(args.save_path)
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(args.save_path):
        print(f"Model already exists in path: {args.save_path}")
        max_acc = torch.load(args.save_path, map_location="cpu")["accuracy"]
    else:
        max_acc = 0
    count = 0
    curr_max = 0

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_count = count_parameters(model)
    print(f"Number of parameters: {params_count}")

    for i in range(epochs):
        train(i, train_loader, model, optimizer, device, scaler)
        acc = eval(i, train_loader, model, device)
        print(f"Epoch {i}: {acc:.4f}")
        if curr_max < acc:
            curr_max = acc
            count = 0
        else:
            count += 1
        if acc > max_acc:
            max_acc = acc
            torch.save(
                {
                    "model_args": model_args,
                    "state_dict": model.state_dict(),
                    "accuracy": max_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"saved to {args.save_path}")
        if count > 10:
            break
