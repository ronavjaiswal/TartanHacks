"""
breath_feedback_pipeline.py

Ultra-stable breath coaching pipeline.
NO Torch. NO Whisper. NO WhisperX. NO pyannote.

Uses:
- ffmpeg (audio extraction)
- Vosk (CPU-only word timestamps)

Safe for FastAPI / Uvicorn / Gunicorn.
"""

from __future__ import annotations
import os
import json
import time
import uuid
import subprocess
from typing import Dict, Any, List, Optional

# ==============================
# Configuration
# ==============================

VOSK_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    "models/vosk-model-small-en-us-0.15"
)

# ==============================
# ffmpeg helpers (ROBUST)
# ==============================

def _wait_for_file_ready(path: str, timeout: float = 3.0):
    """Avoid race conditions on uploaded files."""
    end = time.time() + timeout
    last = -1
    while time.time() < end:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if size == last and size > 0:
            return
        last = size
        time.sleep(0.1)

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    }

def _extract_audio_ffmpeg(input_path: str, out_dir: str, sr: int) -> str:
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input not found: {input_path}")

    os.makedirs(out_dir, exist_ok=True)
    _wait_for_file_ready(input_path)

    out_wav = os.path.join(out_dir, f"_audio_{uuid.uuid4().hex}.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.abspath(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-loglevel", "error",
        os.path.abspath(out_wav),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed\n{e.stderr}\ncmd={' '.join(cmd)}"
        )

    if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
        raise RuntimeError("ffmpeg produced empty audio")

    return out_wav

def pick_audio(input_media: str, out_dir: str, sr: int) -> str:
    return (
        _extract_audio_ffmpeg(input_media, out_dir, sr)
        if _is_video(input_media)
        else input_media
    )

# ==============================
# Breath + bad region parsing
# ==============================

def extract_breath_starts(breath_dict: Dict[str, Any]) -> List[float]:
    return sorted(
        float(b["t_start"])
        for b in breath_dict.get("inhalations", [])
        if "t_start" in b
    )

def extract_bad_regions(bad_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    regions = []
    for label in ("strain", "collapse"):
        for r in bad_dict.get("mistakes", {}).get(label, []):
            regions.append({
                "label": label,
                "start": float(r["start_s"]),
                "end": float(r["end_s"]),
                "peak": float(r["peak"]),
                "note": r.get("note", ""),
            })
    return sorted(regions, key=lambda x: x["start"])

# ==============================
# Vosk transcription (LIGHT)
# ==============================

def transcribe_words_vosk(audio_path: str) -> List[Dict[str, float]]:
    """
    CPU-only word timestamps via Vosk.
    """
    import soundfile as sf
    from vosk import Model, KaldiRecognizer

    if not os.path.exists(VOSK_MODEL_PATH):
        raise RuntimeError(f"Vosk model missing at {VOSK_MODEL_PATH}")

    model = Model(VOSK_MODEL_PATH)

    data, sr = sf.read(audio_path)
    if sr != 16000:
        raise RuntimeError("Audio must be 16kHz mono for Vosk")

    rec = KaldiRecognizer(model, sr)
    rec.SetWords(True)

    words = []
    step = 4000
    for i in range(0, len(data), step):
        rec.AcceptWaveform(data[i:i + step].tobytes())

    res = json.loads(rec.FinalResult())
    for w in res.get("result", []):
        words.append({
            "word": w["word"],
            "start": float(w["start"]),
            "end": float(w["end"]),
        })

    return words

# ==============================
# Alignment helpers
# ==============================

def nearest_before(ts: List[float], t: float) -> Optional[float]:
    c = [x for x in ts if x <= t]
    return max(c) if c else None

def nearest_after(ts: List[float], t: float) -> Optional[float]:
    c = [x for x in ts if x > t]
    return min(c) if c else None

def nearest_word(words, t: float, before=True):
    if before:
        c = [w for w in words if w["start"] <= t]
        return max(c, key=lambda x: x["start"]) if c else None
    else:
        c = [w for w in words if w["start"] > t]
        return min(c, key=lambda x: x["start"]) if c else None

# ==============================
# MAIN ENTRY POINT
# ==============================

def analyze_breaths_and_bad_regions(
    input_media_path: str,
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    out_dir: str = "out_breathcoach",
    sample_rate: int = 16000,
    transcribe: bool = True,
) -> Dict[str, Any]:
    """
    Main pipeline function.
    Safe to call inside FastAPI.
    """

    audio_path = pick_audio(input_media_path, out_dir, sample_rate)

    breaths = extract_breath_starts(breaths_dict)
    bad_regions = extract_bad_regions(bad_dict)

    words = transcribe_words_vosk(audio_path) if transcribe else []

    assignments = []
    for r in bad_regions:
        b = nearest_before(breaths, r["start"])
        relation = "before"
        if b is None:
            b = nearest_after(breaths, r["start"])
            relation = "after"

        w_before = nearest_word(words, b, True) if b else None
        w_after = nearest_word(words, b, False) if b else None

        assignments.append({
            "label": r["label"],
            "region": r,
            "breath_time": b,
            "relation": relation if b else None,
            "word_anchor": {
                "before": w_before,
                "after": w_after,
            },
        })

    coaching_lines = []
    for a in assignments:
        r = a["region"]
        w = a["word_anchor"]["before"]
        word_txt = (
            f"near '{w['word']}' ({w['start']:.2f}s)"
            if w else "near phrase boundary"
        )

        coaching_lines.append(
            f"[{r['label'].upper()}] {r['start']:.2f}–{r['end']:.2f}s | "
            f"breath {a['relation']} {word_txt}. {r['note']}"
        )

    result = {
        "input": input_media_path,
        "audio_used": audio_path,
        "breaths": breaths,
        "bad_regions": bad_regions,
        "words": words,
        "assignments": assignments,
        "coaching_lines": coaching_lines,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "breath_feedback_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
