import json
from typing import List
from pathlib import Path
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_PATH = Path("model/embedding-t/out-of-domain.pkl")

EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if MODEL_PATH.exists():
    CLASSIFIER = joblib.load(MODEL_PATH)
else:
    CLASSIFIER = None
    print(f"[Warning] No classifier found at {MODEL_PATH}")


def detect(file_path: str, tau: float = 0.5) -> List[int]:
    """
    Detect malicious text segments using embedding-t classifier.
    Show progress bar for embedding + prediction.
    """
    if CLASSIFIER is None:
        return []

    ids, texts = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ids.append(data.get("id"))
            texts.append(data.get("text", ""))

    if not texts:
        return []

    X = EMBEDDER.encode(texts, batch_size=32, show_progress_bar=True)

    proba = []
    batch_size = 512
    for i in tqdm(range(0, len(X), batch_size), desc="Predicting", ncols=100):
        batch = X[i:i+batch_size]
        proba_batch = CLASSIFIER.predict_proba(batch)
        proba.append(proba_batch)
    proba = np.vstack(proba)

    idx1 = list(CLASSIFIER.classes_).index(1) if 1 in CLASSIFIER.classes_ else 0
    p = proba[:, idx1]

    preds = (p >= tau).astype(int)

    detect_ids = [id_ for id_, pred in zip(ids, preds) if pred == 1]

    return detect_ids

