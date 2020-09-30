import torch
import torch.nn as nn
import pytorch_lightning as pl
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


class MoADataModule(pl.LightningDataModule):
    def __init__(self, hparams, data, targets):
        super().__init__()
        self.hparams = hparams
        self.data = data
        self.targets = targets

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(self.hparams)
        train_size = self.hparams["train_size"]
        data_tr, data_va, targets_tr, targets_va = train_test_split(
            self.data, self.targets, train_size=train_size, shuffle=False
        )
        self.dataset_tr = MoaDataset(data=data_tr, targets=targets_tr)
        self.dataset_va = MoaDataset(data=data_va, targets=targets_va)

    def train_dataloader(self):
        loader_tr = torch.utils.data.DataLoader(
            self.dataset_tr,
            batch_size=self.hparams["batch_size"],
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        return loader_tr

    def val_dataloader(self):
        loader_va = torch.utils.data.DataLoader(
            self.dataset_va,
            batch_size=self.hparams["batch_size"],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )
        return loader_va

    def test_dataloader(self):
        return None


class LitMoA(pl.LightningModule):
    def __init__(self, hparams, model):
        super(LitMoA, self).__init__()
        self.hparams = hparams
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, threshold=0.00001, mode="min", verbose=True
        )
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "epoch", "monitor": "valid_loss"}],
        )

    def training_step(self, batch, batch_idx):
        data = batch["x"]
        target = batch["y"]
        out = self(data)
        loss = self.criterion(out, target)
        logs = {"training_loss": loss}
        return {"loss": loss, "log": logs, "progress_bar": logs}

    def training_epoch_end(self, outputs):
        loss_avg = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": loss_avg}
        return {"log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        data = batch["x"]
        target = batch["y"]
        out = self(data)
        loss = self.criterion(out, target)
        logs = {"valid_loss": loss}
        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_epoch_end(self, outputs):
        loss_avg = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"valid_loss": loss_avg}
        return {"log": logs, "progress_bar": logs}


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
