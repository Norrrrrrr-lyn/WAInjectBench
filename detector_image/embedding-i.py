#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
from typing import List, Union
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


MODEL_PATH = Path("model/embedding-i/out-of-domain.joblib")

CACHE_ROOT = Path("tmp/embedding_i_cache")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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


def get_cache_root_for_dir(img_dir: Path) -> Path:

    img_dir = img_dir.resolve()
    dir_hash = hashlib.md5(str(img_dir).encode("utf-8")).hexdigest()[:12]
    return CACHE_ROOT / f"{img_dir.name}_{dir_hash}"


def get_embedding_cache_path(img_path: Path, img_dir: Path, cache_dir: Path) -> Path:
    """
    Example:
        img_dir = /data/images
        img_path = /data/images/a/b/1.bmp

        cache path:
        tmp/embedding_i_cache/images_xxx/a/b/1.npy
    """
    rel_path = img_path.resolve().relative_to(img_dir.resolve())
    cache_path = cache_dir / rel_path
    cache_path = cache_path.with_suffix(".npy")
    return cache_path


def list_images(img_dir: Path) -> List[Path]:
    img_files = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    img_files = sorted(img_files)
    return img_files


def extract_one_embedding(img_path: Path) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    image = CLIP_PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = CLIP_MODEL.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    emb_np = emb.cpu().numpy().astype(np.float32).flatten()
    return emb_np


def load_or_compute_embedding(img_path: Path, img_dir: Path, cache_dir: Path) -> Union[np.ndarray, None]:
    cache_path = get_embedding_cache_path(img_path, img_dir, cache_dir)

    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"[Warning] Failed to load cached embedding {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)

    try:
        emb = extract_one_embedding(img_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)

        return emb

    except Exception as e:
        print(f"[Error] Failed to process {img_path}: {e}")
        return None


def parse_image_id(img_path: Path) -> Union[int, str]:
    try:
        return int(img_path.stem)
    except ValueError:
        return img_path.stem


def detect(img_dir: str, tau: float = 0.93) -> List[Union[int, str]]:
    """
    Detect malicious images using CLIP + LogReg classifier.

    Args:
        img_dir: Path to a folder containing images.
                 This function scans images recursively.
        tau: Probability threshold for malicious class.

    Returns:
        List of image IDs predicted as malicious.
        If filename is numeric, returns int.
        Otherwise returns filename stem as str.
    """
    if CLASSIFIER is None:
        return []

    img_dir = Path(img_dir)

    if not img_dir.exists():
        print(f"[Error] Directory not found: {img_dir}")
        return []

    if not img_dir.is_dir():
        print(f"[Error] Not a directory: {img_dir}")
        return []

    img_files = list_images(img_dir)

    if not img_files:
        print(f"[Warning] No images found in {img_dir}")
        return []

    cache_dir = get_cache_root_for_dir(img_dir)

    embeddings = []
    ids = []

    for img_path in tqdm(img_files, desc="Loading/Embedding", ncols=100):
        emb = load_or_compute_embedding(img_path, img_dir, cache_dir)

        if emb is None:
            continue

        embeddings.append(emb)
        ids.append(parse_image_id(img_path))

    if not embeddings:
        return []

    X = np.stack(embeddings).astype(np.float32)

    proba = CLASSIFIER.predict_proba(X)

    if 1 in CLASSIFIER.classes_:
        idx1 = list(CLASSIFIER.classes_).index(1)
    else:
        print("[Warning] Classifier does not contain class 1.")
        return []

    p_malicious = proba[:, idx1]
    preds = (p_malicious >= tau).astype(int)

    detect_ids = [
        img_id for img_id, pred in zip(ids, preds)
        if pred == 1
    ]

    return detect_ids