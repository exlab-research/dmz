import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_train_a", type=str)
    parser.add_argument("--latents_test_a", type=str)
    parser.add_argument("--latents_train_b", type=str)
    parser.add_argument("--latents_test_b", type=str)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers_num", type=int, default=4)
    args = parser.parse_args()
    return args


def parse_data(latents_path):
    loaded = torch.load(latents_path, weights_only=False)
    X = loaded["latents"].float()
    X = X.reshape(X.shape[0], -1, 2)[:, :, 1]
    X = (X > 0.5).float()
    return X


class MappingNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, num_layers):
        super(MappingNetwork, self).__init__()
        layers = list()
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def main():
    args = args_parser()
    print(args)
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    Z_a_train = parse_data(args.latents_train_a)
    Z_a_test = parse_data(args.latents_test_a)
    Z_b_train = parse_data(args.latents_train_b)
    Z_b_test = parse_data(args.latents_test_b)

    dataset_train = TensorDataset(Z_a_train, Z_b_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(Z_a_test, Z_b_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = MappingNetwork(
        input_size=Z_a_train.shape[-1],
        output_size=Z_b_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers_num,
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_inputs, batch_targets in dataloader_train:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            predicted = outputs.round()
            total_correct += (predicted == batch_targets).sum().item()
            total_samples += batch_targets.size(0) * batch_targets.size(1)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader_train):.4f}, Accuracy: {total_correct / total_samples:.4f}"
        )

        with torch.no_grad():
            model.eval()
            total_correct = 0
            total_samples = 0
            for batch_inputs, batch_targets in dataloader_test:
                outputs = model(batch_inputs)
                predicted = outputs.round()  # Convert to 0 or 1 by rounding
                total_correct += (predicted == batch_targets).sum().item()
                total_samples += batch_targets.size(0) * batch_targets.size(1)

            accuracy = total_correct / total_samples
            if best_acc < accuracy:
                best_acc = accuracy
                count = 0
            else:
                count += 1
            print(f"Accuracy: {accuracy * 100:.2f}%")
            if count > 10:
                break

    if args.output_path:
        torch.save({"args": vars(args), "model": model.state_dict(), "acc": accuracy}, args.output_path)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
