import io
from functools import lru_cache

import numpy as np
from PIL import Image

from .config import settings
from .redis_cache import cache_get_bytes, cache_set_bytes, hash_bytes

EMBED_TTL = 30 * 86400  # 30 days


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(settings.clip_model_name)
    processor = CLIPImageProcessor.from_pretrained(settings.clip_model_name)
    model.eval()
    model.to(torch.device(settings.clip_device))
    return model, processor


def embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    import torch

    h = hash_bytes(image_bytes)
    cache_key = f"img:embed:{h}"
    cached = cache_get_bytes(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32).copy()

    model, processor = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.image_embeds  # shape: (1, 512)
        features = features / features.norm(dim=-1, keepdim=True)
    vec = features[0].cpu().numpy().astype(np.float32)
    cache_set_bytes(cache_key, vec.tobytes(), EMBED_TTL)
    return vec


def embed_image_url(url: str) -> np.ndarray:
    import httpx

    response = httpx.get(url, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    return embed_image_bytes(response.content)
