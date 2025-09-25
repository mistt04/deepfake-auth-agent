"""
Video pipeline: sample frames and reuse image analyzer per frame.
"""
import cv2
import numpy as np
from .image_ela import analyze_image_bytes

def analyze_video_bytes(video_bytes: bytes, fps: int = 2):
    # Decode in-memory video using OpenCV
    data = np.frombuffer(video_bytes, dtype=np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(data, cv2.IMREAD_COLOR))
    # Fallback: write to temp if imdecode fails for some containers
    if not cap or not cap.isOpened():
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes); tmp.flush()
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        return {"error": "Unable to open video stream"}, []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = int(max(1, round(native_fps / max(1, fps))))
    frame_idx = 0
    per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Encode frame to JPEG bytes
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                result, _ = analyze_image_bytes(buf.tobytes())
                per_frame.append(result)
        frame_idx += 1
        if len(per_frame) >= 50:  # cap for speed
            break

    if not per_frame:
        return {"error": "No frames analyzed"}, []

    auth_scores = [f["authenticity_score"] for f in per_frame]
    agg = {
        "method": "Frame sampling + ELA baseline",
        "frames_analyzed": len(per_frame),
        "authenticity_score_mean": float(np.mean(auth_scores)),
        "authenticity_score_min": float(np.min(auth_scores)),
        "authenticity_score_max": float(np.max(auth_scores)),
        "per_frame": per_frame[:50]
    }
    return agg, per_frame
