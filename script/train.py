"""This Script is just for training, NOT for inference.
All I need is to get model weight
"""
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import utils
import models

DEVICE = "cuda"
EPOCHS = 3


def objective(trial):
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

    # TODO: ここではロードのみを行いたい。
    # CVのfoldごとにデータをマージするのは面倒
    df = pd.read_csv("input/lish-moa/train_features.csv")
    df = utils.process_data(df)
    folds = pd.read_csv("input/folds/train_folds.csv")

    # Create aux target
    # `nsc_labels` means # of labels found in non-scored train set
    # which is not available in test set.
    non_scored_df = pd.read_csv("input/lish-moa/train_targets_nonscored.csv")
    targets_non_scored = non_scored_df.drop("sig_id", axis=1).to_numpy().sum(axis=1)
    non_scored_df.loc[:, "nsc_labels"] = targets_non_scored
    drop_cols = [c for c in non_scored_df.columns if c not in ("nsc_labels", "sig_id")]
    non_scored_df = non_scored_df.drop(drop_cols, axis=1)
    folds = folds.merge(non_scored_df, on="sig_id", how="left")

    targets = folds.drop(["sig_id", "kfold"], axis=1).columns
    features = df.drop("sig_id", axis=1).columns

    df = df.merge(folds, on="sig_id", how="left")

    print(f"[Fold No.{fold:>3}]\n")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    x_tr = train_df[features].to_numpy()
    x_va = valid_df[features].to_numpy()

    y_tr = train_df[targets].to_numpy()
    y_va = valid_df[targets].to_numpy()

    # TODO: [BEGIN] NN以外の学習を記述

    metrics_callback = utils.MetricsCallback()
    tb_logger = pl_loggers.TensorBoardLogger("log/")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        weights_summary=None,  # "full" for enabling
        callbacks=[metrics_callback],
        progress_bar_refresh_rate=5,
        logger=tb_logger,
    )
    # add param early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc")

    model = utils.LitMoA(
        hparams={},
        model=models.BaseLine(
            num_features=x_tr.shape[1], num_targets=y_tr.shape[1], params=params,
        ),
    )

    dm = utils.MoADataModule(
        hparams={"train_size": x_tr.shape[0], "batch_size": 1024},
        data=np.vstack([x_tr, x_va]).copy(),
        targets=np.vstack([y_tr, y_va]).copy(),
    )

    # Free RAM space as much as possible before training
    del df, train_df, valid_df, x_tr, x_va, y_tr, y_va
    gc.collect()

    trainer.fit(model, dm)

    # TODO: [END] NN以外の学習を記述

    if save_model:
        now = datetime.now()
        now = str(now)[5:17].replace(" ", "_").replace(":", "")
        filename = f"weight/model{now}.pt"
        torch.save(model.model.state_dict(), filename)
        print("model saved at:", filename)

    # optuna needs loss as return
    return metrics_callback.metrics[-1]["valid_loss"].item()


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
        is_pruning = True  # Impl as Param in future

        partial_obj = partial(objective)
        pruner = optuna.pruners.MedianPruner() if is_pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(partial_obj, n_trials=10, timeout=60)

        print("\n---- ---- ---- ---- ----\nBest trial:")
        trial_best = study.best_trial

        print(f"Value: {trial_best.value}")
        print("Params: ")
        best_params = trial_best.params
        print(best_params)
    else:
        with open("config/params_best.json") as f:
            best_params = json.load(f)

    # Training w/o hp tuning
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
            f.write(f"[{now}] best_params: {best_params}, score: {scores}")
    if cv_mode:
        with open("log/cv_score.txt", mode="a") as f:
            f.write(f"[{now}] score: {scores}")
