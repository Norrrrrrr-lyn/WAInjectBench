import os
from typing import List
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip 

MODEL_PATH = Path("model/embedding-i/out-of-domain.joblib")

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
    "ViT-B-32", 
    pretrained="laion2b_s34b_b79k"
)
CLIP_MODEL = CLIP_MODEL.to(device).eval()

if MODEL_PATH.exists():
    CLASSIFIER = joblib.load(MODEL_PATH)
else:
    CLASSIFIER = None
    print(f"[Warning] No classifier found at {MODEL_PATH}")


def detect(img_dir: str, tau: float = 0.5) -> List[int]:
    """
    Detect malicious images using the embedding-i classifier.

    Args:
        img_dir: Path to a folder containing images (named with numbers, e.g. 1.png, 2.jpg).
        tau: Probability threshold for classification (default = 0.5).

    Returns:
        List of image IDs (integers extracted from file names) predicted as malicious.
    """
    if CLASSIFIER is None:
        return []

    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[Error] Directory not found: {img_dir}")
        return []

    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not img_files:
        print(f"[Warning] No images found in {img_dir}")
        return []

    embeddings = []
    ids = []
    for img_path in tqdm(img_files, desc="Embedding", ncols=100):
        try:
            image = CLIP_PREPROCESS(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = CLIP_MODEL.encode_image(image)
            emb = emb.cpu().numpy().flatten()
            embeddings.append(emb)
            img_id = int(img_path.stem)
            ids.append(img_id)

        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {e}")

    if not embeddings:
        return []

    X = np.stack(embeddings)
    proba = CLASSIFIER.predict_proba(X)

    idx1 = list(CLASSIFIER.classes_).index(1) if 1 in CLASSIFIER.classes_ else 0
    p = proba[:, idx1]

    preds = (p >= tau).astype(int)

    detect_ids = [img_id for img_id, pred in zip(ids, preds) if pred == 1]

    return detect_ids
