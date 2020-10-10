"""This Script is just for training, NOT for inference.
All I need is to get model weight
"""
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utils
import models

DEVICE = "cuda"
EPOCHS = 2


def objective(trial):
    # TODO: モデルごとにハイパーパラメータは異なる
    #       モデル内__init__にsuggestを入れるようにしたい
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 7),
        "hidden_size": trial.suggest_int("hidden_size", 16, 2048),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
    }
    loss_all = []
    for fold_ in range(5):
        loss_tmp = run_training(fold_, params, save_model=False)
        loss_all.append(loss_tmp)
    return np.mean(loss_all)


def run_training(fold, params, save_model=False):

    df = pd.read_csv("input/folds/train.csv")
    with open("input/folds/targets", "r") as f:
        targets = f.read().split("\n")
    with open("input/folds/features", "r") as f:
        features = f.read().split("\n")

    print(f"\n[Fold No.{fold:>2}]\n")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    x_tr = train_df[features].to_numpy()
    x_va = valid_df[features].to_numpy()

    y_tr = train_df[targets].to_numpy()
    y_va = valid_df[targets].to_numpy()

    # TODO: [BEGIN] NN以外の学習を記述
    dataset_tr = utils.MoaDataset(x_tr, y_tr)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=1024, num_workers=2)
    dataset_va = utils.MoaDataset(x_va, y_va)
    loader_va = torch.utils.data.DataLoader(dataset_va, batch_size=1024, num_workers=2)

    model = models.BaseLine(num_features=x_tr.shape[1], num_targets=y_tr.shape[1], params=params)
    model.to(DEVICE)

    # TODO: 最適化関数とスケジューラの最適化もoptunaに任せたい
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, threshold=0.00001, mode="min", verbose=True
    )
    eng = utils.Engine(model, optimizer, device=DEVICE)

    # Free RAM space as much as possible before training
    del df, train_df, valid_df, x_tr, x_va, y_tr, y_va
    gc.collect()
    # TODO: [END] NN以外の学習を記述

    loss_best = np.inf
    patience = 10
    patience_cnt = 0
    for ep in range(EPOCHS):
        loss_tr = eng.train(loader_tr)
        loss_va = eng.validate(loader_va)
        scheduler.step(loss_va)
        print(f"epoch:{ep:>2}, train:{loss_tr:>.5}, valid:{loss_va:>.5}")

        if loss_va < loss_best:
            loss_best = loss_va
            if save_model:
                pass
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            break

    print(f"[Fold No.{fold:>2}]")
    print(f"epoch:{ep:>3}, train:{loss_tr:>.5}, valid:{loss_va:>.5}")

    if save_model:
        now = datetime.now()
        now = str(now)[5:17].replace(" ", "_").replace(":", "")
        filename = f"weight/model{now}/fold{fold}.pt"
        torch.save(model.model.state_dict(), filename)
        print("model saved at:", filename)

    return loss_best


if __name__ == "__main__":
    import os
    import argparse
    import json
    import warnings

    import optuna
    from functools import partial

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tune", action="store_true", help="tune hyperparameter if specified"
    )
    parser.add_argument("-v", "--cv", action="store_true", help="cross-validate if specified")
    args = parser.parse_args()
    hp_tune_mode, cv_mode = args.tune, args.cv

    if hp_tune_mode:
        print("Executing hyperparameter tuning...")
        is_pruning = True  # TODO: Impl as Param in future

        partial_obj = partial(objective)
        pruner = optuna.pruners.MedianPruner() if is_pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(partial_obj, n_trials=10, timeout=300)

        print("\n---- ---- ---- ---- ----\nBest trial:")
        trial_best = study.best_trial

        print(f"Value: {trial_best.value}")
        print("Params: ")
        best_params = trial_best.params
        print(best_params)
    else:
        with open("config/params_best.json") as f:
            best_params = json.load(f)

    print("Training with given hyperparameters...")
    _ = run_training(fold=0, params=best_params, save_model=True)

    # Evaluating w/ CV
    if cv_mode:
        scores = 0
        for j in range(5):
            score = run_training(fold=j, params=best_params, save_model=False)
            scores += score
        scores /= 5
        print(f"OOF Score {scores}")

    # Write score record
    now = datetime.now()
    now = str(now)[5:17].replace(" ", "_").replace(":", "")
    if hp_tune_mode:
        with open("log/hp.txt", mode="a") as f:
            # TODO: モデルの名前も出力したい
            # f.write(f"[{now}] name: {<modelname>}, best_params: {best_params}, score: {scores}")
            f.write(f"[{now}] best_params: {best_params}")
    if cv_mode:
        with open("log/cv_score.txt", mode="a") as f:
            f.write(f"[{now}] score: {scores}")
