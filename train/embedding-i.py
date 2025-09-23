import os
import json
import argparse
from pathlib import Path
import torch
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import open_clip


def load_jsonl(file_path):
    paths, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            paths.append(data["path"])
            labels.append(int(data["label"]))
    return paths, labels

def extract_embeddings(image_paths, model, preprocess, device):
    embeddings = []
    for path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = Image.open(path).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(emb.cpu().numpy().flatten())
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            embeddings.append(np.zeros(model.visual.output_dim))
    return np.array(embeddings)

def train_single_classifier(jsonl_file, model, preprocess, device, output_dir, save_emb=False):
    print(f"\nTraining classifier for {jsonl_file} ...")
    image_paths, labels = load_jsonl(jsonl_file)

    embeddings = extract_embeddings(image_paths, model, preprocess, device)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced", 
        n_jobs=-1
    )
    clf.fit(embeddings, labels)

    preds = clf.predict(embeddings)
    print(classification_report(labels, preds))

    model_name = Path(jsonl_file).stem + "_logreg.pkl"
    save_path = os.path.join(output_dir, model_name)
    joblib.dump(clf, save_path)
    print(f"Model saved at {save_path}")

    if save_emb:
        emb_path = os.path.join(output_dir, Path(jsonl_file).stem + "_embeddings.jsonl")
        with open(emb_path, "w", encoding="utf-8") as fout:
            for path, label, emb in zip(image_paths, labels, embeddings):
                entry = {
                    "path": path,
                    "label": label,
                    "embedding": emb.tolist()
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Embeddings saved at {emb_path}")

def main(train_dir, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", 
        pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)


    jsonl_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        print("No jsonl files found in the input directory.")
        return

    for jsonl_file in jsonl_files:
        train_single_classifier(jsonl_file, model, preprocess, device, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression classifiers from image jsonl datasets with CLIP embeddings")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing jsonl training files, jsonl format: \"path\": image path, \"label\": 1/0 (1-malicious, 0-benign)")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save models and embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.device)

