import json
from pathlib import Path
from typing import List
from transformers import pipeline
from tqdm import tqdm

classifier = pipeline("text-classification", model="meta-llama/Llama-Prompt-Guard-2-86M")


def detect(file_path: str) -> List[int]:
    """
    PromptGuard detector for JSONL text files.

    Args:
        file_path (str): path to a JSONL file, where each line is:
                         {"id": int, "text": str}

    Returns:
        List[int]: list of IDs detected as malicious (LABEL_1).
    """
    detect_ids = []

    total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

    with open(file_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total_lines,
                         desc=f"PromptGuard {Path(file_path).name}", ncols=80):
            try:
                item = json.loads(line)
                text_id = item.get("id")
                text = item.get("text", "").strip()

                if text_id is None or not text:
                    continue

                response = classifier(text)
                if response and response[0].get("label") == "LABEL_1":
                    detect_ids.append(text_id)

            except Exception as e:
                print(f"[WARNING] Skipped invalid line in {file_path}: {e}")

    return detect_ids
