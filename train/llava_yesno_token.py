import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def get_yes_no_ids(tokenizer):
    candidates = [(" Yes", " No"), (" yes", " no"), ("Yes", "No"), ("YES", "NO")]
    for y, n in candidates:
        ty = tokenizer(y, add_special_tokens=False).input_ids
        tn = tokenizer(n, add_special_tokens=False).input_ids
        if len(ty) == 1 and len(tn) == 1:
            return ty[0], tn[0], y, n
    raise ValueError("Cannot find single-token verbalizers for Yes/No.")


def _first_param_device(m: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(p for p in m.parameters()).device
    except Exception:
        return fallback


class LlavaYesnoToken(nn.Module):

    def __init__(self, base_model_id: str, dtype: torch.dtype, use_cuda: bool = True):
        super().__init__()
        self.model = LlavaForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=None,
            use_safetensors=True,
        )
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.ID_YES, self.ID_NO, self.VERB_YES, self.VERB_NO = get_yes_no_ids(self.tokenizer)

    def forward(
        self,
        images: Union[List[Image.Image], Image.Image],
        sys_prompt: str,
    ) -> torch.Tensor:

        pil_list = [images] if isinstance(images, Image.Image) else images

        user_text = sys_prompt.strip() + "\nAnswer only with a single token: Yes or No. No explanation."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }]

        batch_prompts = [
            self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            for _ in pil_list
        ]

        inputs = self.processor(
            text=batch_prompts,
            images=pil_list,
            return_tensors="pt",
            padding=True,
            size={"shortest_edge": 280},
        )

        try:
            embed_dev = self.model.get_input_embeddings().weight.device
        except Exception:
            embed_dev = next(self.model.parameters()).device

        vis_dev = embed_dev
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
                vis_dev = _first_param_device(self.model.model.vision_tower, embed_dev)
            elif hasattr(self.model, "vision_tower"):
                vis_dev = _first_param_device(self.model.vision_tower, embed_dev)
        except Exception:
            vis_dev = embed_dev

        if vis_dev.type == "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        for k in ("input_ids", "attention_mask"):
            if k in inputs and isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(embed_dev, non_blocking=True)
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(vis_dev, non_blocking=True)

        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            return_dict=True,
        )
        next_logits = out.logits[:, -1, :]              # [B, vocab]
        two = next_logits[:, [self.ID_NO, self.ID_YES]] # 0:No, 1:Yes
        return two
