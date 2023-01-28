import torch
from diffusers import StableDiffusionPipeline
from config import DEVICE_ID, NUM_IMAGES_PER_PROMPT, NUM_INFERENCE_STEPS, MODEL_PATH, DIFFUSION_PATH
from PIL.Image import Image
from typing import List


pipe = StableDiffusionPipeline.from_pretrained(
    DIFFUSION_PATH, torch_dtype=torch.float16
)
pipe.unet.load_attn_procs(MODEL_PATH)
pipe.to(f"cuda:{DEVICE_ID}")



def generate(prompt: str) -> List[Image]:
    images: List[Image] = pipe(
        prompt=prompt,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS
    ).images
    return images
