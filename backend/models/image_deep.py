import os 
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
import torch 
import torch.nn as nn 
import timm
from torchvision import transforms as T

def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

_DEVICE = _pick_device()
_MODEL = None
_TFMS = None

def _weights_path() -> str:
    env = os.environ.get("DEEPFAKE_WEIGHTS")
    if env and os.path.exists(env):
        return env 
    return os.path.join(os.path.dirname(__file__), "weights", "effnet_deepfake.pth")

def _build_model() -> nn.Module:
    return timm.create_model("efficientnet_b0", pretrained = False, num_classes = 1)

def _load_weights(model: nn.Module, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Deep model weights not found at: {path}. "
            "Set DEEPFAKE_WEIGHTS or place a .pth there."
        )
    state = torch.load(path, map_location="cpu")
    # Handle common checkpoint wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=False)

def _init_model():
    global _MODEL, _TFMS
    if _MODEL is not None:
        return _MODEL, _TFMS

    model = _build_model()
    _load_weights(model, _weights_path())
    model.eval().to(_DEVICE)

    _TFMS = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    _MODEL = model
    return _MODEL, _TFMS

@torch.no_grad()
def analyze_image_bytes(image_bytes: bytes) -> dict:
    model, tfms = _init_model()
    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = tfms(pil).unsqueeze(0).to(_DEVICE)

    logits = model(x)
    logit = logits.flatten()[0]
    prob_fake = torch.sigmoid(logit).item()

    return {
        "authenticity_score": float(1.0 - prob_fake),
        "suspicion_score": float(prob_fake),
        "method": "EfficientNet (deep)",
        "details": {
            "model": "efficientnet_b0",
            "device": str(_DEVICE),
            "prob_fake": float(prob_fake)
        }
    }




