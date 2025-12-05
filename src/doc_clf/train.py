"""Training routines for document-level classifiers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mcc.data import LABEL2ID, MCC_LABELS
from .features import extract_features_for_corpus
from .models import ArgumentRNNClassifier, ArgumentRNNConfig, compute_macro_f1, train_svm


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    if lengths.numel():
        max_len = max(lengths.max().item(), 1)
    else:
        max_len = 1
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def label_to_int(label: str) -> int:
    return 1 if label.lower() == "opinion" else 0


def train_rnn(train_loader, model, optimizer, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for sequences, lengths, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate_rnn(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            logits = model(sequences, lengths)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return compute_macro_f1(all_labels, all_preds)


def parse_args():
    parser = argparse.ArgumentParser(description="Train document classifiers")
    parser.add_argument("--news-train", type=Path, default=Path("data/news_train.jsonl"))
    parser.add_argument("--news-test", type=Path, default=Path("data/news_test.jsonl"))
    parser.add_argument("--mcc-checkpoint", type=Path, default=Path("checkpoints/mcc_bert.pt"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratio_features, sequences, labels = extract_features_for_corpus(
        args.news_train, args.mcc_checkpoint, args.model_name, device
    )
    ratio_features_test, sequences_test, labels_test = extract_features_for_corpus(
        args.news_test, args.mcc_checkpoint, args.model_name, device
    )

    # Train SVM on ratios
    feature_matrix = np.array([[feat[f"ratio_{label}"] for label in MCC_LABELS] for feat in ratio_features])
    svm_labels = [label_to_int(label) for label in labels]
    svm_model, svm_train_f1 = train_svm(feature_matrix, svm_labels)
    feature_matrix_test = np.array(
        [[feat[f"ratio_{label}"] for label in MCC_LABELS] for feat in ratio_features_test]
    )
    svm_labels_test = [label_to_int(label) for label in labels_test]
    svm_preds = svm_model.predict(feature_matrix_test)
    svm_test_f1 = compute_macro_f1(svm_labels_test, svm_preds.tolist())
    print(f"SVM macro-F1 train={svm_train_f1:.3f} test={svm_test_f1:.3f}")

    # Train RNN on sequences
    train_dataset = SequenceDataset(sequences, svm_labels)
    test_dataset = SequenceDataset(sequences_test, svm_labels_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_sequences)

    rnn_config = ArgumentRNNConfig(vocab_size=len(LABEL2ID))
    rnn_model = ArgumentRNNClassifier(rnn_config).to(device)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_rnn(train_loader, rnn_model, optimizer, device)
        f1 = evaluate_rnn(test_loader, rnn_model, device)
        print(f"Epoch {epoch+1} RNN macro-F1: {f1:.3f}")


if __name__ == "__main__":
    train_path = Path("data/news_train.jsonl")
    test_path = Path("data/news_test.jsonl")
    checkpoint_path = Path("checkpoints/mcc_bert.pt")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        train_path.write_text(
            '{"doc_id": "1", "text": "This is news. It reports facts.", "label": "news", "publisher": "Demo"}\n'
            '{"doc_id": "2", "text": "I believe things. This is opinion.", "label": "opinion", "publisher": "Demo"}\n',
            encoding="utf-8",
        )
    if not test_path.exists():
        test_path.write_text(
            '{"doc_id": "3", "text": "Facts happen.", "label": "news", "publisher": "Demo"}\n',
            encoding="utf-8",
        )
    if not checkpoint_path.exists():
        from mcc.models import BertMCCClassifier, MCCModelConfig

        config = MCCModelConfig()
        dummy_model = BertMCCClassifier.from_config(config)
        torch.save(dummy_model.state_dict(), checkpoint_path)
    main()
