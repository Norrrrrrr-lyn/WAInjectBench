
import os
import json
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_jsonl(file_path):
    texts, labels, sources = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
            sources.append(data.get("source", "unknown"))
    return texts, labels, sources

def train_single_classifier(jsonl_file, embedder, output_dir, save_emb=True):
    print(f"Training classifier for {jsonl_file} ...")
    texts, labels, sources = load_jsonl(jsonl_file)

    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, labels)

    preds = clf.predict(embeddings)
    print(classification_report(labels, preds))

    model_name = Path(jsonl_file).stem + "_logreg.pkl"
    save_path = os.path.join(output_dir, model_name)
    joblib.dump(clf, save_path)
    print(f"Model saved at {save_path}")

    if save_emb:
        emb_data = []
        for i in range(len(texts)):
            emb_data.append({
                "embedding": embeddings[i].tolist(),
                "label": labels[i],
                "source": sources[i],
                "text": texts[i]
            })
        emb_path = os.path.join(output_dir, Path(jsonl_file).stem + "_embeddings.jsonl")
        with open(emb_path, "w", encoding="utf-8") as fout:
            for item in emb_data:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Embeddings saved at {emb_path}\n")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        print("No jsonl files found in the input directory.")
        return

    for jsonl_file in jsonl_files:
        train_single_classifier(jsonl_file, embedder, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression classifiers and save embeddings")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder containing jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder to save models and embeddings")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)