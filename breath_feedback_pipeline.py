"""
breath_feedback_pipeline.py

Ultra-stable breath coaching pipeline.
NO transcription.
NO ML.
NO Torch.
NO Whisper.
NO Vosk.

Pure timing-based coaching.
Safe for FastAPI / Uvicorn / Gunicorn.
"""

from __future__ import annotations
import os
import json
import time
import uuid
import subprocess
from typing import Dict, Any, List, Optional

# ============================================================
# ffmpeg utilities (robust + race-safe)
# ============================================================

def _wait_for_file_ready(path: str, timeout: float = 3.0):
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

# ============================================================
# Breath + bad region parsing
# ============================================================

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
                "duration": float(r.get("duration_s", r["end_s"] - r["start_s"])),
                "peak": float(r["peak"]),
                "note": r.get("note", ""),
            })
    return sorted(regions, key=lambda x: x["start"])

# ============================================================
# Timing helpers
# ============================================================

def nearest_before(ts: List[float], t: float) -> Optional[float]:
    c = [x for x in ts if x <= t]
    return max(c) if c else None

def nearest_after(ts: List[float], t: float) -> Optional[float]:
    c = [x for x in ts if x > t]
    return min(c) if c else None

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def analyze_breaths_and_bad_regions(
    input_media_path: str,
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    out_dir: str = "out_breathcoach",
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Main breath coaching analysis.
    NO transcription.
    """

    audio_path = pick_audio(input_media_path, out_dir, sample_rate)

    breaths = extract_breath_starts(breaths_dict)
    bad_regions = extract_bad_regions(bad_dict)

    assignments = []
    coaching_lines = []

    for r in bad_regions:
        b_before = nearest_before(breaths, r["start"])
        b_after = nearest_after(breaths, r["start"])

        if b_before is None and b_after is None:
            advice = "No clear breath nearby — consider adding a reset before this phrase."
            relation = None
            breath_time = None

        elif b_before is None:
            advice = "You entered this phrase without a prep breath. Try breathing earlier."
            relation = "after"
            breath_time = b_after

        elif (r["start"] - b_before) > 2.0:
            advice = "This phrase was long after your last breath. Take a breath closer before it."
            relation = "before"
            breath_time = b_before

        else:
            advice = "Try releasing earlier or lowering intensity near the end of this phrase."
            relation = "before"
            breath_time = b_before

        coaching = (
            f"[{r['label'].upper()}] {r['start']:.2f}–{r['end']:.2f}s | "
            f"{advice}"
        )

        assignments.append({
            "label": r["label"],
            "region": r,
            "breath_time": breath_time,
            "relation": relation,
            "coaching": coaching,
        })

        coaching_lines.append(coaching)

    result = {
        "input": input_media_path,
        "audio_used": audio_path,
        "breaths": breaths,
        "bad_regions": bad_regions,
        "assignments": assignments,
        "coaching_lines": coaching_lines,
        "disclaimer": "Timing-based coaching, not medical advice.",
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "breath_feedback_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
