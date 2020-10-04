from typing import Optional, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import model_parts as parts


class BaseLine(nn.Module):
    def __init__(self, num_features, num_targets, params):
        # num_layers=3, hidden_size=256, dropout=0.3
        super().__init__()
        layers = []
        for _ in range(params["num_layers"]):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, params["hidden_size"], bias=False))
                layers.append(nn.BatchNorm1d(params["hidden_size"]))
                layers.append(nn.Dropout(params["dropout"]))
                nn.PReLU()
            else:
                layers.append(nn.Linear(params["hidden_size"], params["hidden_size"], bias=False))
                layers.append(nn.BatchNorm1d(params["hidden_size"]))
                layers.append(nn.Dropout(params["dropout"]))
                nn.PReLU()
        layers.append(nn.Linear(params["hidden_size"], num_targets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


# DecisionTree
class XGB:
    pass


class LGBM:
    pass


class CatBoost:
    pass


class RGF:
    pass


class RandomForest:
    pass


# Linear
class LogisticRegression:
    pass


class KernelRidge:
    pass


class BayesianRidge:
    pass


class Lasso:
    pass


class ElasticNet:
    pass


class MultitaskLasso:
    pass


class NuSVM:
    pass


class RapidsSVM:
    pass


class OMP:
    """Orthogonal Matching Pursuit."""

    pass


class TabNetSklearn:
    r"""Inspired by https://arxiv.org/pdf/1908.07442.pdf
    Use like skearn learner.
    For stacking, use StackedTabNet"""
    pass


class StackedTabNet:
    r"""Inspired by https://www.kaggle.com/gogo827jz/moa-stacked-tabnet-baseline-tensorflow-2-0"""

    def __init__(self, num_layers):
        pass


# NN


"""
class TemplateNN(nn.Module):
    def __init__(self, num_features, num_targets, params):
        # super().__init__()
        self.model = None

    def forward(self, x):
        x = self.model(x)
        return x
"""


class TReNDSNet10(nn.Module):
    """Inspired by https://www.kaggle.com/c/trends-assessment-prediction/discussion/163017"""

    def __init__(self, num_features, num_targets, params):
        # super().__init__()
        self.model = None

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(ResNet18, self).__init__()
        self.model = parts.ResNet(parts.BasicBlock, [2, 2, 2, 2], num_classes=num_targets)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(ResNet34, self).__init__()
        self.model = parts.ResNet(parts.BasicBlock, [3, 4, 6, 3], num_classes=num_targets)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(ResNet50, self).__init__()
        self.model = parts.ResNet(parts.Bottleneck, [3, 4, 6, 3], num_classes=num_targets)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNext50(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(ResNext50, self).__init__()
        # params["groups"] = 32
        # params["width_per_group"] = 4
        self.model = parts.ResNet(
            parts.Bottleneck, [3, 4, 6, 3], num_classes=num_targets, groups=32, width_per_group=4,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class WideResNet50(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(WideResNet50, self).__init__()
        # params["width_per_group"] = 64*2
        self.model = parts.ResNet(
            parts.Bottleneck, [3, 4, 6, 3], num_classes=num_targets, width_per_group=64 * 2,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_features, num_targets, params):
        super(DenseNet121, self).__init__()
        self.model = parts.DenseNet(
            32, (6, 12, 24, 16), num_init_features=num_features, num_classes=num_targets
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 既存のFWではforward(self, x)以外できないので無理
# TODO: Transformer用のDataLoaderを作る
# TODO: Transformerのshapeを解決する
# nn.Transformer(src: (S, b, E)(S, b, E). tgt: (T, b, E)(T, b, E))
# S = シーケンス長,
# b = バッチサイズ,
# E = 特徴数,
# T = ターゲットのシーケンス長
# class Transformer(nn.Module):
#     """Inspired by https://www.kaggle.com/gogo827jz/moa-lstm-pure-transformer-fast-and-not-bad"""
#     def __init__(self, num_features, num_targets, params):
#         super(Transformer, self).__init__()
#         self.model = nn.Transformer(nhead=16, num_encoder_layers=12)

#     # Requires Target
#     def forward(self, x):
#         out = self.model(x)
#         return out


# class ResNest14d(nn.Module):
#     def __init__(self, num_features, num_targets, params):
#         super(ResNest14d, self).__init__()
#         self.model = parts.ResNet(parts.BasicBlock, [2, 2, 2, 2], num_classes=num_targets)

#     def forward(self, x):
#         x = self.model(x)
#         return x

# TODO: 専用のDataLoaderがあれば使える
# データをグラフに変形する必要があり、作業コスト高め
# class GIN(nn.Module):
#     """Graph isomorphism network"""

#     def __init__(self, num_features, num_targets, params):
#         super(GIN, self).__init__()
#         self.model = parts.GIN(num_features, num_targets, hidden_dim=64,)

#     def forward(self, x, edge_index, batch):
#         x = self.model(x, edge_index, batch)
#         return x
