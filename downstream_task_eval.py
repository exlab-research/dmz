import argparse
import copy
from tqdm import tqdm

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_train", type=str)
    parser.add_argument("--latents_test", type=str)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--auroc", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()
    return args

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, layers):
        super().__init__()
        if layers == 1:
            self.layers = nn.Sequential(nn.Linear(input_dim, num_classes))
        elif layers == 2:
            self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        else:
            assert False, layers

    def forward(self, x):
        return self.layers(x)

def eval_metric(model, laoder, auroc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, ys = list(), list()
    with torch.no_grad():
        for x, y in laoder:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            if auroc:
                preds.append(outputs[:,-1].cpu())
            else:
                preds.append(outputs.argmax(1).cpu())
            ys.append(y.cpu())
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    m = roc_auc_score(ys, preds) if auroc else accuracy_score(ys, preds)
    return m


def perepare_dataset(latents_train_path, latents_test_path, binary):
    train_loaded = torch.load(latents_train_path, map_location="cpu")
    test_loaded = torch.load(latents_test_path, map_location="cpu")

    X_train, y_train = train_loaded["latents"].float(), train_loaded["classes"]
    X_test, y_test = test_loaded["latents"].float(), test_loaded["classes"]

    if len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

    X = torch.cat((X_train, X_test), dim=0)
    y = torch.cat((y_train, y_test), dim=0).long()
    if binary:
        X = (X > 0.5).float()

    return X, y

def eval_lr(X, y, auroc=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    results = list()
    for tr_idx, te_idx in kf.split(X):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        y_train, y_test = y_train.long().ravel(), y_test.long().ravel()
        model = LogisticRegression()
        model.fit(X_train, y_train)

        if auroc:
            test_prob = model.predict_proba(X_test)
            res = roc_auc_score(y_test, test_prob[:,1])
        else:
            test_pred = model.predict(X_test)
            res = accuracy_score(y_test, test_pred)
        results.append(res)
    return np.array(results)

def train_mlp(X, y, lr, batch_size, epochs, layers, auroc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perm = np.random.RandomState(seed=42).permutation(len(X))

    N = int(0.8 * len(X))
    M = int(0.6 * len(X))
    X, y = X[perm], y[perm]
    X_train, y_train, X_val, y_val, X_test, y_test = X[:M], y[:M], X[M:N], y[M:N], X[N:], y[N:]

    num_classes = y_train.max() + 1
    hidden_dim = X_train.shape[-1]

    model = MLP(input_dim=X_train.shape[-1], hidden_dim=hidden_dim, num_classes=num_classes, layers=layers).to(device)

    y_train = y_train.long()
    y_val = y_val.long()
    y_test = y_test.long()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()\

    class_counts = torch.bincount(y_train.long())
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y_train.long()]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader_sampler = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    metric_all = list()
    metric_vals = list()
    pbar = tqdm(range(epochs), total=epochs)

    min_loss = 10000
    count = 0
    best_model = None
    best_val = -1
    for epoch in pbar:
        model.train()
        losses = list()
        for x, y in train_loader_sampler:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        curr_loss = np.mean(losses)
        if curr_loss >= min_loss:
            count += 1
        else:
            count = 0
            min_loss = curr_loss

        model.eval()
        result_train = eval_metric(model, train_loader, auroc=auroc)
        result_val = eval_metric(model, val_loader, auroc=auroc)
        result_test = eval_metric(model, test_loader, auroc=auroc)
        if best_val < result_val:
            best_model = copy.deepcopy(model.state_dict())
            best_val = result_val

        metric_vals.append(result_val)
        metric_all.append((result_train, result_val, result_test))
        metric = "auroc" if auroc else "accuracy"
        pbar.set_description_str(
            f"Epoch {epoch} | Loss train {curr_loss:.4f} | {metric} train: {result_train:.4f} | val: {result_val:.4f} | test: {result_test:.4f}"
        )
        if count > 10:
            break
    idx = np.argmax(metric_vals)
    assert best_val == metric_vals[idx]

    return metric_all[idx], best_model


def main():
    args = args_parser()
    print(args)

    X, y = perepare_dataset(args.latents_train, args.latents_test, args.binary)

    metric = "auroc" if args.auroc else "accuracy"
    results = []
    models_to_save = []
    accs = []
    for feature in range(y.shape[-1]):
        if args.mlp:
            result, model = train_mlp(
                X,
                y[:, feature],
                args.lr,
                args.batch_size,
                args.epochs,
                layers=args.layers,
                auroc=args.auroc,
            )
            result_train, result_val, result_test = result
            print(
                f"MLP-{args.layers}L | {metric} | feature: {feature} | train: {result_train:.4f} | val: {result_val:.4f} | test: {result_test:.4f}"
            )
            results.append(f"MLP-1L,{feature},{metric},{result_train:.4f},{result_val:.4f},{result_test:.4f}\n")
            models_to_save.append(model)
            accs.append(result_test)
        else:
            result = eval_lr(X, y[:,feature], args.auroc)
            results_str = " | ".join([f"{r:.4f}" for r in result])
            print(
                f"LR | {metric} | feature: {feature} | {results_str}"
            )
            results_str = ",".join([f"{r:.4f}" for r in result])
            results.append(f"LR,{feature},{metric},{results_str}")
            accs.append(result)


    if args.mlp:
        final_res = f"AVG: {100*np.mean(accs):.2f}"
    else:
        accs = 100 * np.stack(accs).mean(0)
        final_res = f"AVG: {np.mean(accs):.2f}+{np.std(accs):.2f}"

    print(final_res)

    if args.output_path:
        with open(args.output_path, "w") as f:
            for line in results:
                f.write(line)
            f.write(f"{final_res}\n")
        print(f"Results saved to {args.output_path}")
        if args.mlp:
            path = args.output_path.replace(".csv", "_weights.pt")
            torch.save(models_to_save, path)
            print(f"Weights saved to {path}")



if __name__ == "__main__":
    main()
