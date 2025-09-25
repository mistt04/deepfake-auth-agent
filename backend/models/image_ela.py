"""
Simple, explainable baseline for image manipulation/deepfake cues using
Error Level Analysis (ELA) and high-frequency residual statistics.
"""
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def compute_ela_score(pil_img: Image.Image, quality: int = 95):
    """
    Returns:
        score (float): normalized residual energy [0,1] (higher => more suspicious)
        details (dict): intermediate stats for transparency
        ela_image (PIL.Image): visualization of residuals
    """
    # Recompress to JPEG to expose inconsistencies
    buf = BytesIO()
    pil_img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    # ELA residuals = |original - recompressed|
    ela = ImageChops.difference(pil_img.convert('RGB'), recompressed.convert('RGB'))
    extrema = ela.getextrema()  # per-channel min/max
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela).enhance(scale)

    # Convert to array and compute residual energy
    arr = np.asarray(ela_enhanced).astype(np.float32) / 255.0
    energy = np.mean(arr**2)

    # Normalize a bit heuristically into [0,1]
    # Typical JPEG recompress residuals: ~0.0005 - 0.02
    norm = np.clip((energy - 0.001) / (0.03 - 0.001), 0.0, 1.0)

    details = {
        "jpeg_quality_used": quality,
        "max_channel_diff": float(max_diff),
        "residual_energy": float(energy),
        "scaled_energy": float(norm)
    }
    return float(norm), details, ela_enhanced

def analyze_image_bytes(image_bytes: bytes):
    pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
    score, details, ela_vis = compute_ela_score(pil_img)
    # Provide a naive authenticity score (1 - suspiciousness)
    authenticity = float(1.0 - score)
    return {
        "authenticity_score": authenticity,
        "suspicion_score": float(score),
        "method": "ELA baseline",
        "details": details
    }, ela_vis
