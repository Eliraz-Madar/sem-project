"""Document-level classifiers built on MCC features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn import svm
from sklearn.metrics import f1_score


@dataclass
class ArgumentRNNConfig:
    vocab_size: int
    embed_dim: int = 16
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.1


class ArgumentRNNClassifier(nn.Module):
    """RNN classifier that consumes MCC tag sequences."""

    def __init__(self, config: ArgumentRNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rnn = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(config.hidden_dim, 2)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequences)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        logits = self.classifier(hidden[-1])
        return logits


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    return f1_score(y_true, y_pred, average="macro")


def train_svm(features, labels):
    clf = svm.SVC(kernel="linear")
    clf.fit(features, labels)
    preds = clf.predict(features)
    return clf, compute_macro_f1(labels, preds)


__all__ = ["ArgumentRNNClassifier", "ArgumentRNNConfig", "compute_macro_f1", "train_svm"]
