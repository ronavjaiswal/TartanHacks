"""
breath_feedback_pipeline.py

Stable breath coaching pipeline.
- Robust ffmpeg audio extraction
- Lightweight transcription (plain Whisper)
- Breath → bad region → word anchoring
- No WhisperX, no pyannote, no torch serialization issues

Author: YOU
"""

from __future__ import annotations
import os
import re
import json
import math
import time
import uuid
import subprocess
from typing import Any, Dict, List, Optional

# ============================================================
# ffmpeg utilities (robust + concurrent-safe)
# ============================================================

def _wait_for_file_stable(path: str, timeout_s: float = 3.0, poll_s: float = 0.1):
    """Wait until file size stops changing (upload race protection)."""
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(poll_s)
            continue
        if size == last and size > 0:
            return
        last = size
        time.sleep(poll_s)

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    }

def _run_ffmpeg_extract_audio(input_path: str, out_dir: str, sr: int = 16000) -> str:
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input file not found: {input_path}")

    os.makedirs(out_dir, exist_ok=True)
    _wait_for_file_stable(input_path)

    out_wav = os.path.join(out_dir, f"_extracted_{uuid.uuid4().hex}.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.abspath(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-fflags", "+genpts",
        "-loglevel", "error",
        os.path.abspath(out_wav),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ffmpeg audio extraction failed\n"
            f"stderr:\n{e.stderr}\n"
            f"cmd: {' '.join(cmd)}"
        )

    if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
        raise RuntimeError("ffmpeg produced empty wav")

    return out_wav

def _pick_audio_path(input_path: str, out_dir: str, sr: int) -> str:
    return _run_ffmpeg_extract_audio(input_path, out_dir, sr) if _is_video(input_path) else input_path

# ============================================================
# Breath + bad region parsing
# ============================================================

def extract_breaths(breaths_dict: Dict[str, Any]) -> List[float]:
    """Only breath START times matter."""
    return sorted(
        float(b["t_start"])
        for b in breaths_dict.get("inhalations", [])
        if "t_start" in b
    )

def extract_bad_regions(bad_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for label in ("strain", "collapse"):
        for r in bad_dict.get("mistakes", {}).get(label, []):
            out.append({
                "label": label,
                "start": float(r["start_s"]),
                "end": float(r["end_s"]),
                "peak": float(r["peak"]),
                "note": r.get("note", ""),
            })
    return sorted(out, key=lambda x: x["start"])

# ============================================================
# Lightweight transcription (plain Whisper)
# ============================================================

_WHISPER_MODEL = None

def transcribe_words(audio_path: str, device: str = "cpu") -> List[Dict[str, float]]:
    """
    Plain Whisper transcription.
    Word timestamps are approximated (spread across segments).
    """
    global _WHISPER_MODEL
    try:
        import whisper
    except ImportError:
        return []

    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("small", device=device)

    res = _WHISPER_MODEL.transcribe(audio_path, fp16=(device != "cpu"))
    segments = res.get("segments", []) or []

    words = []
    for seg in segments:
        s0 = float(seg["start"])
        s1 = float(seg["end"])
        text = seg.get("text", "").strip()
        if not text or s1 <= s0:
            continue

        toks = [t for t in re.split(r"\s+", text) if t]
        step = (s1 - s0) / max(len(toks), 1)

        for i, tok in enumerate(toks):
            words.append({
                "word": tok,
                "start": s0 + i * step,
                "end": min(s1, s0 + (i + 1) * step),
            })

    return words

# ============================================================
# Mapping logic
# ============================================================

def nearest_before(times: List[float], t: float) -> Optional[float]:
    b = [x for x in times if x <= t]
    return max(b) if b else None

def nearest_after(times: List[float], t: float) -> Optional[float]:
    a = [x for x in times if x > t]
    return min(a) if a else None

def nearest_word(words, t, direction="before"):
    if direction == "before":
        c = [w for w in words if w["start"] <= t]
        return max(c, key=lambda x: x["start"]) if c else None
    else:
        c = [w for w in words if w["start"] > t]
        return min(c, key=lambda x: x["start"]) if c else None

# ============================================================
# Main public API
# ============================================================

def analyze_breaths_and_bad_regions(
    input_media_path: str,
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    out_dir: str = "out_breathcoach",
    sample_rate: int = 16000,
    transcribe: bool = True,
) -> Dict[str, Any]:

    audio_used = _pick_audio_path(input_media_path, out_dir, sample_rate)

    breaths = extract_breaths(breaths_dict)
    regions = extract_bad_regions(bad_dict)

    words = transcribe_words(audio_used) if transcribe else []

    assignments = []
    for r in regions:
        b = nearest_before(breaths, r["start"])
        if b is None:
            b = nearest_after(breaths, r["start"])
            rel = "after"
        else:
            rel = "before"

        wb = nearest_word(words, b, "before") if b and words else None
        wa = nearest_word(words, b, "after") if b and words else None

        assignments.append({
            "label": r["label"],
            "region": r,
            "breath_start": b,
            "breath_relation": rel if b else None,
            "word_anchor": {"before": wb, "after": wa},
        })

    coaching_lines = []
    for a in assignments:
        r = a["region"]
        b = a["breath_start"]
        w = a["word_anchor"]["before"]
        word_txt = f"near '{w['word']}' ({w['start']:.2f}s)" if w else "near phrase boundary"

        coaching_lines.append(
            f"[{r['label'].upper()}] {r['start']:.2f}–{r['end']:.2f}s | "
            f"breath {a['breath_relation']} at {b:.2f}s {word_txt}. "
            f"{r['note']}"
        )

    result = {
        "input": input_media_path,
        "audio_used": audio_used,
        "breaths": breaths,
        "bad_regions": regions,
        "words": words,
        "assignments": assignments,
        "coaching_lines": coaching_lines,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "breath_feedback_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
