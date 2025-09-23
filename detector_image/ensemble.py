import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def detect(result_dir: str) -> List[Dict]:
    """
    Ensemble detector:
    - Reads all JSONL results from detectors in result_dir
    - Combines detect_ids by union
    - Recomputes TPR/FPR
    - Returns a list of results (same format as other detectors)
    """
    data_map = defaultdict(lambda: {"detect_ids": set(), "total_num": 0, "is_malicious": None})

    for file in Path(result_dir).glob("*.jsonl"):
        if file.name == "ensemble.jsonl":  # skip old ensemble outputs
            continue
        with open(file, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    entry = json.loads(line)
                    data_name = entry["data_name"]

                    data_map[data_name]["detect_ids"].update(entry.get("detect_ids", []))

                    if data_map[data_name]["total_num"] == 0:
                        data_map[data_name]["total_num"] = entry.get("total_num", 0)

                    if "tpr" in entry:
                        data_map[data_name]["is_malicious"] = True
                    elif "fpr" in entry:
                        data_map[data_name]["is_malicious"] = False
                except Exception as e:
                    print(f"[WARN] Failed to parse line in {file}: {e}")

    results = []
    for data_name, info in data_map.items():
        detect_ids = list(info["detect_ids"])
        total_num = info["total_num"]
        is_malicious = info["is_malicious"]

        if total_num > 0:
            if is_malicious:
                rate_key, rate_value = "tpr", round(len(detect_ids) / total_num, 4)
            else:
                rate_key, rate_value = "fpr", round(len(detect_ids) / total_num, 4)
        else:
            rate_key, rate_value = "tpr", 0.0 if is_malicious else "fpr", 0.0  # 保底

        result = {
            "data_name": data_name,
            rate_key: rate_value,
            "detect_ids": detect_ids,
            "total_num": total_num,
        }
        results.append(result)

    return results

