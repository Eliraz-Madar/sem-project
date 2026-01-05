"""Convert dataset.csv to JSONL format for Part B document classification."""
import csv
import json
import random
from pathlib import Path


def convert_csv_to_jsonl(csv_path, train_output, test_output, split_ratio=0.8, shuffle=True, random_seed=42):
    """
    Convert dataset.csv to JSONL format required by Part B.
    
    Expected CSV format:
        ID, category, article_content
    
    Output JSONL format:
        {"doc_id": "...", "text": "...", "label": "...", "publisher": "Unknown"}
    
    Args:
        csv_path: Path to the dataset.csv file
        train_output: Path to save training JSONL
        test_output: Path to save test JSONL
        split_ratio: Percentage of data to use for training (default 0.8)
        shuffle: Whether to shuffle data before splitting (default True)
        random_seed: Random seed for reproducible shuffling (default 42)
    """
    csv_path = Path(csv_path)
    train_output = Path(train_output)
    test_output = Path(test_output)
    
    # Create output directories if they don't exist
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    documents = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert to JSONL format
            # The label needs to be either "fact" or "opinion" (lowercase)
            # Map "fact" -> "news" or keep as is based on what Part B expects
            doc = {
                "doc_id": row['ID'],
                "text": row['article_content'],
                "label": row['category'].lower(),  # Ensure lowercase (fact/opinion)
                "publisher": "Unknown"
            }
            documents.append(doc)
    
    print(f"Total documents loaded: {len(documents)}")
    
    # Shuffle documents to ensure balanced train/test split
    if shuffle:
        random.seed(random_seed)
        random.shuffle(documents)
        print(f"Data shuffled with random seed: {random_seed}")
    
    # Split into train and test
    split_index = int(len(documents) * split_ratio)
    train_docs = documents[:split_index]
    test_docs = documents[split_index:]
    
    print(f"Training documents: {len(train_docs)}")
    print(f"Test documents: {len(test_docs)}")
    
    # Write training JSONL
    with open(train_output, 'w', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Write test JSONL
    with open(test_output, 'w', encoding='utf-8') as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"\nConversion complete!")
    print(f"Training data saved to: {train_output}")
    print(f"Test data saved to: {test_output}")
    
    # Print label distribution
    train_labels = [doc['label'] for doc in train_docs]
    test_labels = [doc['label'] for doc in test_docs]
    
    print(f"\nTraining label distribution:")
    for label in set(train_labels):
        count = train_labels.count(label)
        print(f"  {label}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    print(f"\nTest label distribution:")
    for label in set(test_labels):
        count = test_labels.count(label)
        print(f"  {label}: {count} ({count/len(test_labels)*100:.1f}%)")


if __name__ == "__main__":
    # Default paths
    csv_file = Path(__file__).parent.parent / "dataset.csv"
    train_jsonl = Path(__file__).parent.parent / "src" / "data" / "news_train.jsonl"
    test_jsonl = Path(__file__).parent.parent / "src" / "data" / "news_test.jsonl"
    
    print(f"Converting {csv_file}")
    print(f"Output training: {train_jsonl}")
    print(f"Output test: {test_jsonl}")
    print("-" * 60)
    
    convert_csv_to_jsonl(csv_file, train_jsonl, test_jsonl, split_ratio=0.8)
