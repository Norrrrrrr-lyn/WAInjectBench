import os
import shutil
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import spacy

# Add JailGuard utils to path
import sys
sys.path.append('YOUR_PATH_TO_/JailGuard/JailGuard/utils')
from utils import read_file_list, update_divergence, detect_attack
from mask_utils import load_mask_dir
from augmentations import img_aug_dict
from minigpt_utils import initialize_model, model_inference


def get_method(method_name):
    try:
        return img_aug_dict[method_name]
    except KeyError:
        raise ValueError(f"Unknown augmentation method: {method_name}")


def load_and_convert_image(image_path: str) -> Image.Image:
    pil_img = Image.open(image_path)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    return pil_img


def test_single_image(
    image_path: str,
    question_text: str,
    vis_processor,
    chat,
    model,
    mutator="PL",
    number=8,
    threshold=0.025,
    tag="default"
):
    """
    Test a single image with JailGuard.

    Steps:
    1. Generate multiple augmented variants of the input image.
    2. Run inference on each variant with the vision-language model.
    3. Analyze the diversity of responses to detect possible jailbreaks.

    Args:
        image_path (str): Path to the input image.
        question_text (str): The text prompt to ask the model.
        vis_processor: Visual processor from MiniGPT.
        chat: Chat interface.
        model: The main vision-language model.
        mutator (str): Image augmentation method.
        number (int): Number of variants to generate.
        threshold (float): Divergence threshold for detection.
        tag (str): A unique tag to name temp directories.

    Returns:
        bool: True if the image is detected as an attack, False otherwise.
    """
    variant_save_dir = f"./temp/temp_variants_{tag}"
    response_save_dir = f"./temp/temp_responses_{tag}"
    for d in [variant_save_dir, response_save_dir]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    pil_img = load_and_convert_image(image_path)
    image_format = "jpg" if image_path.lower().endswith(".jpg") else "bmp"
    for i in range(number):
        tmp_method = get_method(mutator)
        new_image = tmp_method(img=pil_img)
        new_image.save(os.path.join(variant_save_dir, f"{i}-{mutator}.{image_format}"))

    with open(os.path.join(variant_save_dir, "question"), "w") as f:
        f.write(question_text)

    variant_list, name_list = load_mask_dir(variant_save_dir)
    for j in range(len(variant_list)):
        img_prompt_path = variant_list[j]
        prompts_eval = [question_text, img_prompt_path]
        save_name = name_list[j].split(".")[0]
        new_save_path = os.path.join(response_save_dir, save_name)

        if not os.path.exists(new_save_path):
            result = model_inference(vis_processor, chat, model, prompts_eval)
            with open(new_save_path, "w", encoding="utf-8") as f:
                f.write(result)

    metric = spacy.load("en_core_web_md")
    check_list = [os.path.join(response_save_dir, f) for f in os.listdir(response_save_dir)]
    output_list = read_file_list(check_list)
    max_div, jailbreak_keywords = update_divergence(
        output_list, "test_image", response_save_dir, select_number=number, metric=metric, top_string=100
    )
    detection_result = detect_attack(max_div, jailbreak_keywords, threshold)

    shutil.rmtree(variant_save_dir)
    shutil.rmtree(response_save_dir)

    return detection_result


print("Initializing JailGuard model...")
vis_processor, chat, model = initialize_model()


def detect(folder_path: str) -> List[int]:
    """
    Args:
        folder_path (str): Path to a folder containing images.

    Returns:
        List[int]: List of image IDs (integers from file names) detected as malicious.
    """
    folder = Path(folder_path)
    detect_ids = []

    for img_file in tqdm(list(folder.glob("*")), desc=f"JailGuard {folder.name}", ncols=80):
        if not img_file.is_file():
            continue
        try:
            is_attack = test_single_image(
                image_path=str(img_file),
                question_text="Describe this image.",
                vis_processor=vis_processor,
                chat=chat,
                model=model,
                mutator="PL",
                number=8,
                threshold=0.025,
                tag=folder.name
            )
            if is_attack:
                try:
                    img_id = int(img_file.stem)
                    detect_ids.append(img_id)
                except ValueError:
                    print(f"[WARNING] Non-numeric filename skipped: {img_file.name}")
        except Exception as e:
            print(f"[WARNING] Failed on {img_file}: {e}")

    return detect_ids

