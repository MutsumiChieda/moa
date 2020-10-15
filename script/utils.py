import torch
import torch.nn as nn

import pandas as pd
from sklearn.model_selection import train_test_split


class MoaDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.data[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float),
        }


class TestMoaDataset:
    """ dataset for moa competition.
    Usage:
        dataset = TestMoADataset(dataset=test_features.iloc[:, 1:].values)
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.dataset[item, :], dtype=torch.float),
        }


def add_dummies(data, col):
    enc = pd.get_dummies(data[col])
    oh_cols = [f"{col}_{c}" for c in enc.columns]
    enc.columns = oh_cols
    data = data.drop(col, axis=1)
    data = data.join(enc)
    return data


def process_data(df):
    df = add_dummies(df, "cp_time")
    df = add_dummies(df, "cp_dose")
    df = add_dummies(df, "cp_type")
    return df


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, dataloader):
        self.model.train()
        loss_final = 0
        for data in dataloader:
            self.optimizer.zero_grad()
            inputs = data["x"].to(self.device, non_blocking=True)
            targets = data["y"].to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            loss_final += loss.item()
        return loss_final / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        loss_final = 0
        for data in dataloader:
            inputs = data["x"].to(self.device, non_blocking=True)
            targets = data["y"].to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss_final += loss.item()
        return loss_final / len(dataloader)
