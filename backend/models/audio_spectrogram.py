"""
Audio baseline: use mel-spectrogram features with simple heuristics
(spectral flatness, zero-crossing rate). This is NOT a robust detector,
but demonstrates the pipeline.
"""
import io
import numpy as np
import soundfile as sf
import librosa

def analyze_audio_bytes(audio_bytes: bytes, sr: int = 16000):
    data, in_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:  # mixdown stereo
        data = np.mean(data, axis=1)
    if in_sr != sr:
        data = librosa.resample(data, orig_sr=in_sr, target_sr=sr)

    # Features
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=data)))
    S = np.abs(librosa.stft(data, n_fft=1024, hop_length=256)) + 1e-9
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_stats = {
        "mel_min_db": float(np.min(mel_db)),
        "mel_mean_db": float(np.mean(mel_db)),
        "mel_max_db": float(np.max(mel_db))
    }

    # Naive heuristic: overly flat spectrum & odd zcr can be suspicious
    suspicion = np.clip((flatness - 0.2) / 0.6, 0, 1) * 0.7 + np.clip((zcr - 0.05) / 0.2, 0, 1) * 0.3
    authenticity = float(1.0 - suspicion)

    return {
        "method": "Spectrogram heuristics (flatness + ZCR)",
        "authenticity_score": authenticity,
        "suspicion_score": float(suspicion),
        "features": {
            "spectral_flatness": flatness,
            "zero_crossing_rate": zcr,
            **mel_stats
        }
    }
