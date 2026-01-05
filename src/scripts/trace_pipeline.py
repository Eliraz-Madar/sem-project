"""End-to-end pipeline tracer with detailed logging and validation.

This script verifies the complete seminar spec pipeline:
- STAGE A: Sentence-level argument mining (claim/premise/other classification)
- STAGE B: Document-level feature aggregation (ratios + sequences)
- STAGE C: Document classifier training and evaluation (SVM + RNN)

Usage:
    python -m scripts.trace_pipeline --help
    python -m scripts.trace_pipeline --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_webis_tiny.pt
    python -m scripts.trace_pipeline --smoke  # Quick test with 20 docs
    python -m scripts.trace_pipeline --dry-run  # Check files without running
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score

from mcc.data import MCC_LABELS, LABEL2ID
from mcc.models import BertMCCClassifier, MCCModelConfig
from doc_clf.features import extract_features_for_corpus, load_documents
from doc_clf.models import train_svm, compute_macro_f1, ArgumentRNNClassifier, ArgumentRNNConfig
from doc_clf.train import SequenceDataset, collate_sequences, label_to_int, train_rnn, evaluate_rnn
from torch.utils.data import DataLoader


# ============================================================================
# Timing & Logging Utilities
# ============================================================================

@dataclass
class StageMetrics:
    """Metrics collected during a pipeline stage."""
    stage_name: str
    elapsed_seconds: float
    start_timestamp: str
    end_timestamp: str
    metadata: Dict[str, Any]


class PipelineTracer:
    """Collects timing and metrics across the entire pipeline."""
    
    def __init__(self):
        self.start_time = time.perf_counter()
        self.stage_metrics: List[StageMetrics] = []
        self.global_start = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def elapsed(self) -> float:
        """Total elapsed time since tracer creation."""
        return time.perf_counter() - self.start_time
    
    def record_stage(self, stage_name: str, elapsed: float, metadata: Dict[str, Any]):
        """Record metrics for a completed stage."""
        self.stage_metrics.append(StageMetrics(
            stage_name=stage_name,
            elapsed_seconds=elapsed,
            start_timestamp=metadata.get("start_timestamp", ""),
            end_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
        ))
    
    def save_report(self, output_path: Path):
        """Save trace report to JSON."""
        report = {
            "pipeline_start": self.global_start,
            "pipeline_end": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_seconds": self.elapsed(),
            "stages": [asdict(stage) for stage in self.stage_metrics],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Trace report saved to: {output_path}")


@contextmanager
def timed_block(tracer: PipelineTracer, stage_name: str, description: str = ""):
    """Context manager for timing and logging a pipeline stage."""
    banner = "=" * 80
    print(f"\n{banner}")
    print(f"  {stage_name}")
    if description:
        print(f"  {description}")
    print(banner)
    print(f"⏱  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱  Elapsed since start: {tracer.elapsed():.2f}s\n")
    
    stage_start = time.perf_counter()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata = {"start_timestamp": start_timestamp}
    
    try:
        yield metadata
    finally:
        stage_elapsed = time.perf_counter() - stage_start
        print(f"\n⏱  Stage completed in: {stage_elapsed:.2f}s")
        print(f"⏱  Total elapsed: {tracer.elapsed():.2f}s")
        tracer.record_stage(stage_name, stage_elapsed, metadata)


# ============================================================================
# Pipeline Stages
# ============================================================================

def verify_files(args, tracer: PipelineTracer) -> bool:
    """Verify all required files exist and are readable."""
    with timed_block(tracer, "FILE VERIFICATION", "Checking input files and dependencies"):
        files_to_check = [
            (args.news_train, "Training data"),
            (args.news_test, "Test data"),
            (args.mcc_checkpoint, "MCC checkpoint"),
        ]
        
        all_ok = True
        for file_path, description in files_to_check:
            if not file_path.exists():
                print(f"❌ {description} NOT FOUND: {file_path}")
                print(f"   → Suggestion: Check the path or run data preparation scripts.")
                all_ok = False
            else:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✓ {description}: {file_path} ({size_mb:.2f} MB)")
        
        # Check if MCC checkpoint is loadable
        if args.mcc_checkpoint.exists():
            try:
                checkpoint = torch.load(args.mcc_checkpoint, map_location="cpu")
                print(f"✓ MCC checkpoint loadable ({len(checkpoint)} state dict keys)")
            except Exception as e:
                print(f"❌ MCC checkpoint corrupted or incompatible: {e}")
                print(f"   → Suggestion: Re-train the MCC model with: python -m scripts.run_mcc_training")
                all_ok = False
        
        return all_ok


def load_and_sample_data(args, tracer: PipelineTracer) -> Tuple[List[dict], List[dict]]:
    """Load training and test documents, optionally sampling for smoke test."""
    with timed_block(tracer, "DATA LOADING", "Loading and preparing datasets") as meta:
        print(f"📂 Loading training data from: {args.news_train}")
        train_docs = load_documents(args.news_train)
        print(f"   → Loaded {len(train_docs)} training documents")
        
        print(f"📂 Loading test data from: {args.news_test}")
        test_docs = load_documents(args.news_test)
        print(f"   → Loaded {len(test_docs)} test documents")
        
        # Check for publisher diversity (cross-domain split)
        train_publishers = set(doc.get("publisher", "unknown") for doc in train_docs)
        test_publishers = set(doc.get("publisher", "unknown") for doc in test_docs)
        overlap = train_publishers & test_publishers
        
        print(f"\n📊 Publisher Analysis:")
        print(f"   Train publishers: {sorted(train_publishers)}")
        print(f"   Test publishers: {sorted(test_publishers)}")
        if overlap:
            print(f"   ⚠  Publisher overlap detected: {sorted(overlap)}")
            print(f"      (Cross-domain split NOT fully implemented)")
        else:
            print(f"   ✓ No publisher overlap (cross-domain split)")
        
        # Smoke test sampling
        if args.smoke:
            print(f"\n🔥 SMOKE TEST MODE: Sampling {args.smoke_size} documents from each set")
            train_docs = train_docs[:args.smoke_size]
            test_docs = test_docs[:args.smoke_size]
            print(f"   → Sampled {len(train_docs)} train, {len(test_docs)} test")
        
        meta["train_count"] = len(train_docs)
        meta["test_count"] = len(test_docs)
        meta["train_publishers"] = sorted(train_publishers)
        meta["test_publishers"] = sorted(test_publishers)
        
        return train_docs, test_docs


def run_stage_a(args, tracer: PipelineTracer, device: torch.device) -> Tuple[BertMCCClassifier, Any]:
    """STAGE A: Load MCC model and verify it works."""
    with timed_block(tracer, "STAGE A: SENTENCE-LEVEL ARGUMENT MINING", 
                     "Loading BERT-based MCC classifier (claim/premise/other)") as meta:
        
        print(f"🤖 Loading MCC model: {args.model_name}")
        config = MCCModelConfig(pretrained_model_name=args.model_name)
        model = BertMCCClassifier.from_config(config)
        
        print(f"📦 Loading checkpoint: {args.mcc_checkpoint}")
        checkpoint = torch.load(args.mcc_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        print(f"✓ Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Num labels: {config.num_labels}")
        print(f"   Labels: {MCC_LABELS}")
        
        # Quick sanity check
        test_sentences = [
            "The sky is blue.",
            "This proves my point.",
            "Hello world.",
        ]
        print(f"\n🧪 Sanity check with {len(test_sentences)} test sentences:")
        encoded = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        
        for sent, pred in zip(test_sentences, preds):
            print(f"   '{sent}' → {MCC_LABELS[pred]}")
        
        meta["model_name"] = args.model_name
        meta["checkpoint"] = str(args.mcc_checkpoint)
        meta["device"] = str(device)
        
        return model, tokenizer


def run_stage_b(args, tracer: PipelineTracer, train_docs: List[dict], test_docs: List[dict],
                device: torch.device) -> Tuple[List, List, List, List, List, List]:
    """STAGE B: Extract document-level features from MCC predictions."""
    with timed_block(tracer, "STAGE B: DOCUMENT-LEVEL FEATURE AGGREGATION",
                     "Extracting ratio features and tag sequences") as meta:
        
        print(f"🔍 Processing training documents ({len(train_docs)} docs)...")
        ratio_features_train, sequences_train, labels_train = extract_features_for_corpus(
            args.news_train, args.mcc_checkpoint, args.model_name, device
        )
        
        print(f"\n🔍 Processing test documents ({len(test_docs)} docs)...")
        ratio_features_test, sequences_test, labels_test = extract_features_for_corpus(
            args.news_test, args.mcc_checkpoint, args.model_name, device
        )
        
        # Print sample features for first 3 docs
        print(f"\n📊 Sample Features (first 3 training documents):")
        for i in range(min(3, len(train_docs))):
            doc = train_docs[i]
            features = ratio_features_train[i]
            seq = sequences_train[i]
            label = labels_train[i]
            
            print(f"\n  Doc {i+1}: '{doc.get('doc_id', f'doc_{i}')}' (Label: {label})")
            print(f"    Text preview: {doc['text'][:100]}...")
            
            # Print first 5 sentence predictions
            from doc_clf.features import split_into_sentences
            sentences = split_into_sentences(doc['text'])
            print(f"    Sentences ({len(sentences)} total, showing first 5):")
            for j, (sent, tag) in enumerate(zip(sentences[:5], seq[:5])):
                print(f"      [{j+1}] {MCC_LABELS[tag]}: '{sent[:60]}...'")
            
            # Print aggregated features
            print(f"    Aggregated features:")
            for k, v in features.items():
                print(f"      {k}: {v:.3f}")
        
        # Check if positional features are implemented
        if "first_claim_position" in ratio_features_train[0]:
            print(f"\n✓ Positional features detected (e.g., first_claim_position)")
        else:
            print(f"\n⚠  Positional features NOT IMPLEMENTED")
            print(f"   (Only ratio features available: {list(ratio_features_train[0].keys())})")
        
        meta["train_samples"] = len(ratio_features_train)
        meta["test_samples"] = len(ratio_features_test)
        meta["feature_names"] = list(ratio_features_train[0].keys())
        meta["avg_sequence_length_train"] = np.mean([len(s) for s in sequences_train])
        meta["avg_sequence_length_test"] = np.mean([len(s) for s in sequences_test])
        
        return ratio_features_train, sequences_train, labels_train, ratio_features_test, sequences_test, labels_test


def run_stage_c(args, tracer: PipelineTracer,
                ratio_features_train, sequences_train, labels_train,
                ratio_features_test, sequences_test, labels_test,
                device: torch.device) -> Dict[str, Any]:
    """STAGE C: Train and evaluate document classifiers."""
    results = {}
    
    # Convert labels to integers
    y_train = [label_to_int(label) for label in labels_train]
    y_test = [label_to_int(label) for label in labels_test]
    
    # ========== BASELINE: SVM on Ratio Features ==========
    with timed_block(tracer, "STAGE C.1: BASELINE LEXICAL MODEL (SVM)",
                     "Training SVM classifier on argument component ratios") as meta:
        
        print(f"📐 Preparing feature matrices...")
        X_train = np.array([[feat[f"ratio_{label}"] for label in MCC_LABELS] for feat in ratio_features_train])
        X_test = np.array([[feat[f"ratio_{label}"] for label in MCC_LABELS] for feat in ratio_features_test])
        
        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")
        print(f"   Features: {[f'ratio_{label}' for label in MCC_LABELS]}")
        
        print(f"\n🤖 Training SVM (linear kernel)...")
        svm_model, svm_train_f1 = train_svm(X_train, y_train)
        
        print(f"\n📊 Evaluating on test set...")
        y_pred_svm = svm_model.predict(X_test)
        svm_test_f1 = compute_macro_f1(y_test, y_pred_svm.tolist())
        svm_test_acc = accuracy_score(y_test, y_pred_svm)
        
        cm = confusion_matrix(y_test, y_pred_svm)
        print(f"\n📈 SVM Results:")
        print(f"   Train Macro-F1: {svm_train_f1:.4f}")
        print(f"   Test Macro-F1:  {svm_test_f1:.4f}")
        print(f"   Test Accuracy:  {svm_test_acc:.4f}")
        print(f"\n   Confusion Matrix (rows=true, cols=pred):")
        print(f"             news  opinion")
        print(f"   news    {cm[0][0]:5d}  {cm[0][1]:7d}")
        print(f"   opinion {cm[1][0]:5d}  {cm[1][1]:7d}")
        
        meta["train_f1"] = svm_train_f1
        meta["test_f1"] = svm_test_f1
        meta["test_accuracy"] = svm_test_acc
        meta["confusion_matrix"] = cm.tolist()
        
        results["svm"] = {
            "train_f1": svm_train_f1,
            "test_f1": svm_test_f1,
            "test_accuracy": svm_test_acc,
            "confusion_matrix": cm.tolist(),
        }
    
    # ========== ARGUMENTATION MODEL: RNN on Sequences ==========
    with timed_block(tracer, "STAGE C.2: ARGUMENTATION-BASED MODEL (RNN)",
                     "Training GRU classifier on argument component sequences") as meta:
        
        print(f"📦 Preparing sequence datasets...")
        train_dataset = SequenceDataset(sequences_train, y_train)
        test_dataset = SequenceDataset(sequences_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_sequences)
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        print(f"\n🤖 Initializing RNN model...")
        rnn_config = ArgumentRNNConfig(vocab_size=len(LABEL2ID))
        rnn_model = ArgumentRNNClassifier(rnn_config).to(device)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=args.lr)
        
        print(f"   Architecture: GRU (embed_dim={rnn_config.embed_dim}, hidden_dim={rnn_config.hidden_dim})")
        print(f"   Vocab size: {rnn_config.vocab_size}")
        print(f"   Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
        
        epoch_f1s = []
        for epoch in range(args.epochs):
            print(f"\n  Epoch {epoch+1}/{args.epochs}")
            train_rnn(train_loader, rnn_model, optimizer, device)
            f1 = evaluate_rnn(test_loader, rnn_model, device)
            epoch_f1s.append(f1)
            print(f"    ✓ Test F1: {f1:.4f}")
        
        final_f1 = epoch_f1s[-1] if epoch_f1s else 0.0
        
        print(f"\n📈 RNN Results:")
        print(f"   Final Test Macro-F1: {final_f1:.4f}")
        print(f"   F1 progression: {' → '.join([f'{f:.3f}' for f in epoch_f1s])}")
        
        meta["final_test_f1"] = final_f1
        meta["epoch_f1s"] = epoch_f1s
        meta["architecture"] = "GRU"
        
        results["rnn"] = {
            "final_test_f1": final_f1,
            "epoch_f1s": epoch_f1s,
        }
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline tracer with detailed logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default paths
  python -m scripts.trace_pipeline

  # Quick smoke test with 20 documents
  python -m scripts.trace_pipeline --smoke

  # Dry-run to check files only
  python -m scripts.trace_pipeline --dry-run

  # Custom paths
  python -m scripts.trace_pipeline \\
      --news-train data/news_train.jsonl \\
      --news-test data/news_test.jsonl \\
      --mcc-checkpoint checkpoints/mcc_bert.pt \\
      --out-dir outputs/trace_run_001
        """
    )
    
    # Input files
    parser.add_argument("--news-train", type=Path, default=Path("data/news_train.jsonl"),
                        help="Training documents (JSONL)")
    parser.add_argument("--news-test", type=Path, default=Path("data/news_test.jsonl"),
                        help="Test documents (JSONL)")
    parser.add_argument("--mcc-checkpoint", type=Path, default=Path("checkpoints/mcc_webis_tiny.pt"),
                        help="Pre-trained MCC model checkpoint")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                        help="BERT model name for MCC")
    
    # Output
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/trace"),
                        help="Directory for trace reports and logs")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for RNN training")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for RNN training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for RNN")
    
    # Modes
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode: use only first N documents for quick validation")
    parser.add_argument("--smoke-size", type=int, default=20,
                        help="Number of documents to use in smoke test mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: only check file existence, don't run pipeline")
    
    return parser.parse_args()


def main():
    args = parse_args()
    tracer = PipelineTracer()
    
    print("=" * 80)
    print("  SEMINAR PROJECT: PIPELINE TRACER")
    print("  End-to-end validation of argument-based news classification")
    print("=" * 80)
    print(f"📅 Started at: {tracer.global_start}")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if args.smoke:
        print(f"🔥 Mode: SMOKE TEST (first {args.smoke_size} docs only)")
    elif args.dry_run:
        print(f"🌵 Mode: DRY RUN (file checks only)")
    else:
        print(f"🚀 Mode: FULL PIPELINE")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # File verification
    if not verify_files(args, tracer):
        print("\n❌ File verification failed. Aborting.")
        return 1
    
    if args.dry_run:
        print("\n✓ Dry-run complete. All files found.")
        return 0
    
    # Load data
    train_docs, test_docs = load_and_sample_data(args, tracer)
    
    # STAGE A: Load MCC model
    mcc_model, tokenizer = run_stage_a(args, tracer, device)
    
    # STAGE B: Extract features
    ratio_features_train, sequences_train, labels_train, \
    ratio_features_test, sequences_test, labels_test = run_stage_b(
        args, tracer, train_docs, test_docs, device
    )
    
    # STAGE C: Train classifiers
    results = run_stage_c(
        args, tracer,
        ratio_features_train, sequences_train, labels_train,
        ratio_features_test, sequences_test, labels_test,
        device
    )
    
    # Final summary
    with timed_block(tracer, "PIPELINE SUMMARY", "Generating final report"):
        print(f"\n📊 Overall Results:")
        print(f"   SVM (Baseline):  Test F1 = {results['svm']['test_f1']:.4f}, Acc = {results['svm']['test_accuracy']:.4f}")
        print(f"   RNN (Argument):  Test F1 = {results['rnn']['final_test_f1']:.4f}")
        
        print(f"\n💾 Saving outputs to: {args.out_dir}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trace report
        report_path = args.out_dir / "trace_report.json"
        tracer.save_report(report_path)
        
        # Save results
        results_path = args.out_dir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_path}")
        
        print(f"\n🎉 Pipeline complete! Total time: {tracer.elapsed():.2f}s")
    
    return 0


if __name__ == "__main__":
    exit(main())
