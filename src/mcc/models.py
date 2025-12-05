"""Sentence-level BERT classifier for argument component detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoConfig

from .data import ID2LABEL


@dataclass
class MCCModelConfig:
    """Configuration bundle that drives :class:`BertMCCClassifier`."""

    pretrained_model_name: str = "bert-base-uncased"
    num_labels: int = 3
    dropout: float = 0.1


class BertMCCClassifier(nn.Module):
    """Light wrapper around ``AutoModelForSequenceClassification``.

    The wrapper only exists so we can pin the ``id2label``/``label2id`` mappings
    and expose a friendlier ``from_config`` constructor.
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    @classmethod
    def from_config(cls, config: MCCModelConfig) -> "BertMCCClassifier":
        return cls(config.pretrained_model_name, config.num_labels)

    def forward(self, **batch) -> Dict[str, torch.Tensor]:  # pragma: no cover - exercised via HF tests
        return self.model(**batch)


__all__ = ["MCCModelConfig", "BertMCCClassifier"]
