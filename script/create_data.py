import sys
from os.path import exists
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append("script/")
import utils
import models


def split_data():
    print("Split data")
    path_fold = "input/folds/train_folds.csv"
    if not exists(path_fold):
        df = pd.read_csv("input/lish-moa/train_targets_scored.csv")
        df.loc[:, "kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        targets = df.drop("sig_id", axis=1).values

        mskf = MultilabelStratifiedKFold(n_splits=5)
        for fold_, (tr_, va_) in enumerate(mskf.split(X=df, y=targets)):
            df.loc[va_, "kfold"] = fold_
        df.to_csv(path_fold, index=False)
        print(f"Created: {path_fold}")
    else:
        print("Skipped: already exists")


def preprocess():
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
    df.to_csv("input/folds/train.csv", index=False)

    # Serialize column names
    with open("input/folds/targets", "w") as f:
        f.write("\n".join(targets))
    with open("input/folds/features", "w") as f:
        f.write("\n".join(features))


if __name__ == "__main__":
    split_data()
    preprocess()
