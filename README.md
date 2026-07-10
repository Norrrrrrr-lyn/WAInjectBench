# WAInjectBench

<!-- <p align="center">
  <img src="paper_figure.png" alt="WAInjectBench Overview" width="600"/>
</p> -->

WAInjectBench is a comprehensive benchmark for **prompt injection detection** in web agents.  
It covers **6 types of attacks**, across **two modalities**: **text** and **image**.

---

## 📂 Dataset Structure

- `data/`
  - `text/`
    - `benign/` → 4 categories, stored as JSONL files
    - `malicious/` → 8 attack types, stored as JSONL files
  - `image/`
    - `benign/` → 2 categories, stored in subfolders
    - `malicious/` → 7 attack types, stored in subfolders

---

## ⚙️ Installation

Clone the repo and create the environment:

```bash
git clone https://github.com/Norrrrrrr-lyn/WAInjectBench.git
cd WAInjectBench
conda env create -f environment.yml
conda activate wainjectbench
```

## 🚀 Evaluation

WAInjectBench supports two evaluation pipelines: text-based detection and image-based detection.

### 🔹 Text-based Detection

```bash
python main_text.py \
  --data_dir [path to text dataset] \
  --detector [detector name] \
  --result_dir [output path] \
  --gpu [gpu id]
```

Available detectors:
["kad", "promptarmor", "embedding-t", "promptguard", "datasentinel", "ensemble"]

PromptArmor → requires OPENAI_API_KEY as environment variable.

DataSentinel →

```bash
git clone https://github.com/liu00222/Open-Prompt-Injection.git
```

Download the pretrained model into: WAInjectBench/Open-Prompt-Injection/DataSentinel_Models
Set the directory and model path in detector_text/datasentinel.py.

### 🔹 Image-based Detection

```bash
python main_image.py \
  --data_dir [path to image dataset] \
  --detector [detector name] \
  --result_dir [output path] \
  --gpu [gpu id]
```

Available detectors:
["gpt-4o-prompt", "llava-1.5-7b-prompt", "jailguard", "embedding-i", "llava-1.5-7b-ft", "ensemble"]

GPT-4o-Prompt → requires OPENAI_API_KEY as environment variable.

JailGuard →
```bash
git clone https://github.com/shiningrain/JailGuard.git
```

Follow its README to configure MiniGPT4.

LLaVA-1.5-7B-FT → requires downloading our finetuned model and setting its path in detector_image/llava.py.

### 🔹 In-domain Generalization

We also provide in-domain trained versions of the Embedding-T and Embedding-I models, available in model/embedding-t/in-domain and model/embedding-i/in-domain. To use them, follow the same evaluation procedure as in the main experiments, but update the model path in detector_text/embedding-t.py and detector_image/embedding-i.py.

## 🏋️ Training

We provide code for training embedding-based binary classifiers for both text and image.

Text embedding classifier

```bash
python train/embedding-t.py \
  --input_dir [dir with training text jsonl files] \
  --output_dir [model output path]
```

JSONL format:
```bash
{"text": "example", "label": 1}   # 1 for malicious, 0 for benign
```

Image embedding classifier

```bash
python train/embedding-i.py \
  --input_dir [dir with training image jsonl files] \
  --output_dir [model output path]
```

JSONL format:
```bash
{"path": "path/to/image.png", "label": 1}
```

Finetuning LLaVA-1.5-7B

```bash
python train.py \
  --train_jsonl train.jsonl \
  --val_jsonl val.jsonl \
  --use_lora \
  --amp_dtype bf16 \
  --device_mode single \
  --gpu_id 0
```

The JSONL files should contain image paths and labels (1 = malicious, 0 = benign).
Experiments in our paper use the default hyperparameters.
