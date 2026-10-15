"""Train the three models.

Protocol (matches the paper): for every epoch budget E in 1..10 and trial t in 1..3, a fresh
model is trained from scratch for E epochs (Adam, lr 0.001, batch size 128) and saved as
models/<label>_epEE_tT.pth -- 90 models in total. Each model is then evaluated on the clean
MNIST test set, and its training time, accuracy and loss are logged.

The log is written to results/training_results.csv. An existing log is never overwritten
unless --overwrite is passed; use --log to write somewhere else.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_CLASSES, MODEL_DIR, MODEL_LABELS, TRAINING_CSV,
                    TRIALS, evaluate, get_device, model_path, test_tensors, train_dataset)


def run_training_session(label, epochs, trial, device, x_test, y_test):
    loader = DataLoader(train_dataset(label), batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=device.type == "cuda")
    model = MODEL_CLASSES[label]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = round(time.perf_counter() - start_time, 4)

    torch.save(model.state_dict(), model_path(label, epochs, trial))
    acc, test_loss, _ = evaluate(model, x_test, y_test)   # evaluate() switches to eval mode
    return duration, round(acc, 2), round(test_loss, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODEL_LABELS)
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--trials", nargs="+", type=int, default=TRIALS)
    p.add_argument("--log", default=TRAINING_CSV)
    p.add_argument("--overwrite", action="store_true", help="replace an existing log file and existing model files")
    args = p.parse_args()

    if os.path.exists(args.log) and not args.overwrite:
        raise SystemExit(f"{args.log} already exists; pass --overwrite or choose another --log path")
    existing = [model_path(m, e, t) for m in args.models for e in args.epochs for t in args.trials
                if os.path.exists(model_path(m, e, t))]
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} model files already exist in {MODEL_DIR}; pass --overwrite to replace them")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerow(["Model_Type", "Epochs", "Trial", "Duration_Seconds", "Accuracy", "Test_Loss"])

    device = get_device()
    x_test, y_test = test_tensors(device)
    for label in args.models:
        for ep in args.epochs:
            for t in args.trials:
                duration, acc, loss = run_training_session(label, ep, t, device, x_test, y_test)
                with open(args.log, "a", newline="") as f:
                    csv.writer(f).writerow([label, ep, t, duration, acc, loss])
                print(f"Saved: {label} | Ep: {ep} | Trial: {t} | Time: {duration}s | Acc: {acc}% | Loss: {loss}")

    print(f"\nTraining complete | Log: {args.log} | Models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
