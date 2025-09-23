import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

def load_detector(detector_name: str):
    """
    Dynamically load the corresponding detector module.
    - For most detectors: import detector_image.{detector_name}
    - For LLaVA series (llava-1.5-7b-prompt / llava-1.5-7b-ft): always import detector_image.llava_detect
    """
    import importlib

    if detector_name in ["llava-1.5-7b-prompt", "llava-1.5-7b-ft"]:
        module = importlib.import_module("detector_image.llava")
        return module

    try:
        module = importlib.import_module(f"detector_image.{detector_name}")
        return module
    except ImportError:
        raise ValueError(
            f"Detector {detector_name} not found. "
            f"Make sure you have a file under detector_image/ named {detector_name}.py"
        )



def process_folder(folder_path: Path, detector, detector_name: str, is_malicious: bool) -> Dict:
    data_name = folder_path.name

    if detector_name in ["llava-1.5-7b-prompt", "llava-1.5-7b-ft"]:
        from detector_image import llava
        detect_files = llava.detect(str(folder_path), detector_name=detector_name)
    else:
        detect_files = detector.detect(str(folder_path))

    detect_ids = [int(f) for f in detect_files]
    total_num = len(list(folder_path.glob("*")))

    if is_malicious:
        rate_key, rate_value = "tpr", round(len(detect_ids) / total_num, 4) if total_num > 0 else 0.0
    else:
        rate_key, rate_value = "fpr", round(len(detect_ids) / total_num, 4) if total_num > 0 else 0.0

    return {
        "data_name": data_name,
        rate_key: rate_value,
        "detect_ids": detect_ids,
        "total_num": total_num,
    }



def run_experiment(data_dir: str, detector_name: str, result_dir: str, gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if detector_name == "ensemble":
        from detector_image import ensemble
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
        parent_path = data_dir / folder_name
        if not parent_path.exists():
            continue
        for sub_folder in parent_path.iterdir():
            if sub_folder.is_dir():
                res = process_folder(sub_folder, detector, detector_name, is_malicious=(folder_name == "malicious"))
                results.append(res)

    output_path = result_dir / f"{detector_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in results:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Experiment Framework for Image Detectors")

    parser.add_argument("--data_dir", type=str, default="data/image",
                        help="Input data directory containing benign/ and malicious/ folders")
    parser.add_argument("--detector", type=str, required=True,
                        choices=["gpt-4o-prompt", "llava-1.5-7b-prompt", "jailguard",
                                 "embedding-i", "llava-1.5-7b-ft", "ensemble"],
                        help="Detector to use")
    parser.add_argument("--result_dir", type=str, default="result/image",
                        help="Directory to store results")
    parser.add_argument("--gpu", type=str, default="0",
                        help="Which GPU to use (e.g., '0', '1', '0,1')")

    args = parser.parse_args()

    run_experiment(args.data_dir, args.detector, args.result_dir, args.gpu)