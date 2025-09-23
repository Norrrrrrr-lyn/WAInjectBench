import os
import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_detector(detector_name: str):
    """
    Dynamically load the corresponding detector module.
    Each detector module must implement a `detect(file_path)` function,
    which takes a JSONL file path as input and outputs a list of detected IDs.
    """
    import importlib
    try:
        module = importlib.import_module(f"detector_text.{detector_name}")
        return module
    except ImportError:
        raise ValueError(f"Detector {detector_name} not found. "
                         f"Make sure you have a file under detector_text/ named {detector_name}.py")


def process_file(file_path: Path, detector, is_malicious: bool) -> Dict:
    """
    file_path: path to the JSONL file
    detector: loaded detector module
    is_malicious: whether the file comes from the malicious folder
    """
    data_name = file_path.name
    detect_ids = detector.detect(str(file_path))

    total_num = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

    if is_malicious:
        rate_key, rate_value = "tpr", round(len(detect_ids) / total_num, 4) if total_num > 0 else 0.0
    else:
        rate_key, rate_value = "fpr", round(len(detect_ids) / total_num, 4) if total_num > 0 else 0.0

    result = {
        "data_name": data_name,
        rate_key: rate_value,
        "detect_ids": detect_ids,
        "total_num": total_num,
    }

    return result


def run_experiment(data_dir: str, detector_name: str, result_dir: str, gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if detector_name == "ensemble":
        from detector_text import ensemble
        results = ensemble.detect(result_dir)
        output_path = Path(result_dir) / "ensemble.jsonl"
        with open(output_path, "w", encoding="utf-8") as fout:
            for entry in results:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Ensemble results saved to {output_path}")
        return

    detector = load_detector(detector_name)

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for folder_name in ["benign", "malicious"]:
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            continue
        for file in folder_path.glob("*.jsonl"):
            res = process_file(file, detector, is_malicious=(folder_name == "malicious"))
            results.append(res)

    output_path = result_dir / f"{detector_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in results:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/text",
                        help="Input data directory containing benign/ and malicious/ folders")
    parser.add_argument("--detector", type=str, required=True,
                        choices=["kad", "promptarmor", "embedding-t", "promptguard", "datasentinel", "ensemble"],
                        help="Detector to use")
    parser.add_argument("--result_dir", type=str, default="result/text",
                        help="Directory to store results")
    parser.add_argument("--gpu", type=str, default="9",
                        help="Which GPU to use (e.g., '0', '1', '0,1')")

    args = parser.parse_args()

    run_experiment(args.data_dir, args.detector, args.result_dir, args.gpu)
