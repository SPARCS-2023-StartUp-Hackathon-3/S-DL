import torch
from diffusers import CycleDiffusionPipeline
from ..config import DEVICE_ID, NUM_IMAGES_PER_PROMPT, NUM_INFERENCE_STEPS, MODEL_PATH
from PIL.Image import Image
from typing import List


pipe = CycleDiffusionPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to(f"cuda:{DEVICE_ID}")


def edit(source_image: Image, source_prompt: str, target_prompt: str) -> List[Image]:
    images: List[Image] = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        image=source_image,
        num_inference_steps=NUM_INFERENCE_STEPS
    ).images
    return images
