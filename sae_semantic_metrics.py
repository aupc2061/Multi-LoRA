from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils import get_prompt, load_lora_info


@dataclass(frozen=True)
class LoRASemanticSpec:
    lora_id: str
    name: str
    triggers: list[str]
    category: str
    generic_prompt: str
    full_prompt: str
    trigger_sentence: str
    prompt_variants: list[str]


def build_lora_semantic_spec(image_style: str, lora_info_path: str, lora_id: str) -> LoRASemanticSpec:
    lora_info = load_lora_info(image_style, lora_info_path)
    generic_prompt, _ = get_prompt(image_style)

    for category, group in lora_info.items():
        for lora in group:
            if lora["id"] != lora_id:
                continue

            triggers = [str(token).strip() for token in lora.get("trigger", []) if str(token).strip()]
            if not triggers:
                raise ValueError(f"LoRA {lora_id} has no trigger phrases in metadata.")

            full_prompt = generic_prompt + ", " + ", ".join(triggers)
            trigger_sentence = ", ".join(triggers)
            prompt_variants = [full_prompt, trigger_sentence, *triggers]
            return LoRASemanticSpec(
                lora_id=lora_id,
                name=str(lora.get("name", lora_id)),
                triggers=triggers,
                category=category,
                generic_prompt=generic_prompt,
                full_prompt=full_prompt,
                trigger_sentence=trigger_sentence,
                prompt_variants=prompt_variants,
            )

    raise ValueError(f"LoRA id not found in metadata: {lora_id}")


class CLIPSemanticScorer:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def score_image_text(self, images: Iterable[Image.Image], texts: list[str]) -> list[float]:
        image_list = [image.convert("RGB") for image in images]
        if len(image_list) != len(texts):
            raise ValueError("Number of images and texts must match.")

        inputs = self.processor(text=texts, images=image_list, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarities = (image_embeds * text_embeds).sum(dim=-1)
        return [float(score.item()) for score in similarities.cpu()]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_ablation_semantics(
    scorer: CLIPSemanticScorer,
    spec: LoRASemanticSpec,
    base_image: Image.Image,
    edited_image: Image.Image,
) -> dict[str, float]:
    base_generic = scorer.score_image_text([base_image], [spec.generic_prompt])[0]
    edited_generic = scorer.score_image_text([edited_image], [spec.generic_prompt])[0]

    base_full = scorer.score_image_text([base_image], [spec.full_prompt])[0]
    edited_full = scorer.score_image_text([edited_image], [spec.full_prompt])[0]

    base_trigger_scores = scorer.score_image_text([base_image] * len(spec.prompt_variants), spec.prompt_variants)
    edited_trigger_scores = scorer.score_image_text([edited_image] * len(spec.prompt_variants), spec.prompt_variants)

    base_trigger_mean = _mean(base_trigger_scores)
    edited_trigger_mean = _mean(edited_trigger_scores)

    generic_drop = base_generic - edited_generic
    full_prompt_drop = base_full - edited_full
    trigger_mean_drop = base_trigger_mean - edited_trigger_mean

    return {
        "clip_generic_base": base_generic,
        "clip_generic_intervened": edited_generic,
        "clip_generic_drop": generic_drop,
        "clip_full_prompt_base": base_full,
        "clip_full_prompt_intervened": edited_full,
        "clip_full_prompt_drop": full_prompt_drop,
        "clip_trigger_mean_base": base_trigger_mean,
        "clip_trigger_mean_intervened": edited_trigger_mean,
        "clip_trigger_mean_drop": trigger_mean_drop,
        "clip_semantic_specificity": trigger_mean_drop - generic_drop,
    }
