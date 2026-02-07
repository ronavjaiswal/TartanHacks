#!/usr/bin/env python3
"""
breath_feedback_pipeline.py

Single-file pipeline that:
1) Extracts audio from an input MP4 via ffmpeg
2) Transcribes audio with word-level timestamps (WhisperX; free/local)
3) Maps "inhale start" timestamps to nearby words
4) Assigns bad regions (strain/collapse) to nearby breaths (usually breath before)
5) Optionally calls an OpenAI LLM to generate human coaching lines
6) Writes outputs to out_dir

USAGE (CLI testing):
  python breath_feedback_pipeline.py \
    --mp4 input.mp4 \
    --breaths_json breaths.json \
    --bad_json bad.json \
    --out_dir out \
    --do_llm

Where:
- breaths.json contains your big breath dict (with "inhalations": [{"t_start": ...}, ...])
- bad.json contains:
    {"strain":[{"start_s":..., "end_s":..., "peak":...}, ...],
     "collapse":[{"start_s":..., "end_s":..., "peak":...}, ...]}

INSTALL:
  pip install -U numpy
  pip install -U whisperx
  pip install -U openai   # only if using --do_llm

SYSTEM:
  ffmpeg must be installed and on PATH.

NOTES:
- Breath durations are not trusted; we use inhale START times.
- WhisperX provides word-level timestamps via alignment (free/local).
"""

import os
import json
import argparse
import subprocess
from typing import Dict, List, Any, Optional, Tuple

OPENAI_API_KEY = NotImplemented

# ----------------------------
# Audio extraction
# ----------------------------
def extract_audio_ffmpeg(mp4_path: str, wav_out: str, sr: int = 16000) -> str:
    os.makedirs(os.path.dirname(wav_out), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "wav", wav_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_out


# ----------------------------
# WhisperX transcription
# ----------------------------
def transcribe_words_whisperx(audio_path: str, device: str = "cpu", model_size: str = "small") -> List[Dict[str, float]]:
    """
    Returns list of word dicts: [{"word": str, "start": float, "end": float}, ...]
    """
    import whisperx  # type: ignore

    model = whisperx.load_model(model_size, device=device)
    result = model.transcribe(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device=device)

    words: List[Dict[str, float]] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({"word": str(w["word"]), "start": float(w["start"]), "end": float(w["end"])})
    return words


# ----------------------------
# Breath parsing
# ----------------------------
def breaths_from_pose_dict(breath_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your reliable timestamps are inhalations[*].t_start.
    We convert to a uniform list of breath events.
    """
    out: List[Dict[str, Any]] = []
    inhalations = breath_dict.get("inhalations", []) or []
    for inh in inhalations:
        try:
            t_start = float(inh["t_start"])
        except Exception:
            continue
        t_end = float(inh.get("t_end", t_start))  # unreliable but included
        out.append({
            "start": t_start,
            "end": t_end,
            "confidence": float(inh.get("confidence", 0.5)) if "confidence" in inh else 0.5,
            "sources": ["pose", "audio"],  # keep your existing convention
        })
    out.sort(key=lambda b: b["start"])
    return out


# ----------------------------
# Word lookup helpers
# ----------------------------
def nearest_word_before(words: List[Dict[str, float]], t: float) -> Optional[Dict[str, float]]:
    cand = [w for w in words if w["start"] <= t]
    return max(cand, key=lambda w: w["start"]) if cand else None

def nearest_word_after(words: List[Dict[str, float]], t: float) -> Optional[Dict[str, float]]:
    cand = [w for w in words if w["start"] > t]
    return min(cand, key=lambda w: w["start"]) if cand else None

def attach_words_to_breaths(
    breaths: List[Dict[str, Any]],
    words: List[Dict[str, float]],
    max_gap_s: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    For each breath start, attach word_before/word_after if within max_gap_s.
    """
    for b in breaths:
        t = float(b["start"])
        w_prev = nearest_word_before(words, t)
        w_next = nearest_word_after(words, t)

        if w_prev and (t - float(w_prev["start"])) <= max_gap_s:
            b["word_before"] = w_prev
        else:
            b["word_before"] = None

        if w_next and (float(w_next["start"]) - t) <= max_gap_s:
            b["word_after"] = w_next
        else:
            b["word_after"] = None

    return breaths


# ----------------------------
# Assign bad regions -> breaths
# ----------------------------
def _breath_before(breaths: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    b = [x for x in breaths if float(x["start"]) <= t]
    return max(b, key=lambda x: float(x["start"])) if b else None

def _breath_after(breaths: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    b = [x for x in breaths if float(x["start"]) > t]
    return min(b, key=lambda x: float(x["start"])) if b else None

def assign_bad_to_breaths(
    breaths: List[Dict[str, Any]],
    bad: Dict[str, List[Dict[str, Any]]],
    pre_window_s: float = 10.0,
    post_window_s: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    For each bad region, choose a breath before it if close enough,
    else choose a breath after it if close enough.
    """
    assignments: List[Dict[str, Any]] = []

    for label in ("strain", "collapse"):
        regions = bad.get(label, []) or []
        for r in regions:
            rs = float(r["start_s"])
            re = float(r.get("end_s", rs))
            peak = float(r.get("peak", 0.0))

            prev_b = _breath_before(breaths, rs)
            next_b = _breath_after(breaths, rs)

            chosen, rel = None, None
            if prev_b is not None and (rs - float(prev_b["start"])) <= pre_window_s:
                chosen, rel = prev_b, "before"
            elif next_b is not None and (float(next_b["start"]) - rs) <= post_window_s:
                chosen, rel = next_b, "after"
            else:
                chosen = prev_b or next_b
                rel = "before" if chosen is prev_b else ("after" if chosen is next_b else None)

            assignments.append({
                "label": label,
                "region_start": rs,
                "region_end": re,
                "peak": peak,
                "breath": chosen,
                "breath_relation": rel,
            })

    assignments.sort(key=lambda a: float(a["region_start"]))
    return assignments


# ----------------------------
# Build structured alignment payload
# ----------------------------
def build_alignment_payload(
    mp4_path: str,
    audio_path: str,
    breaths: List[Dict[str, Any]],
    words: List[Dict[str, float]],
    bad_assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    enriched: List[Dict[str, Any]] = []
    for a in bad_assignments:
        b = a.get("breath", None)
        if b is None:
            enriched.append({**a, "breath_start": None, "anchor_word": None, "anchor_word_time": None})
            continue

        breath_start = float(b["start"])
        anchor = b.get("word_before") or b.get("word_after")

        enriched.append({
            "label": a["label"],
            "region_start": float(a["region_start"]),
            "region_end": float(a["region_end"]),
            "peak": float(a["peak"]),
            "breath_relation": a.get("breath_relation"),
            "breath_start": breath_start,
            "anchor_word": (anchor["word"] if anchor else None),
            "anchor_word_time": (float(anchor["start"]) if anchor else None),
        })

    return {
        "mp4": mp4_path,
        "audio": audio_path,
        "breaths": breaths,  # includes word_before/word_after
        "words_count": len(words),
        "bad_assignments": enriched,
    }


# ----------------------------
# OpenAI LLM feedback (optional)
# ----------------------------
def llm_generate_feedback_openai(
    alignment_payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> str:
    """
    Requires: pip install -U openai and OPENAI_API_KEY env var.
    """
    from openai import OpenAI  # type: ignore
    client = OpenAI()

    # Keep it focused: only the mapped events (top few already)
    events = alignment_payload.get("bad_assignments", [])

    prompt = {
        "task": "Generate concise breath/coaching feedback for a singer.",
        "rules": [
            "Only comment on STRAIN and PHRASE COLLAPSE events.",
            "Reference breath_start timestamp and anchor_word when available.",
            "If anchor_word missing, reference only the timestamp.",
            "Give actionable advice: where to breathe earlier/later, reset support, lighten onset, avoid pushing.",
            "Return 3–8 bullet points max.",
            "Do not invent words or timestamps beyond the provided data."
            "If you are able to figure out which song it is, and are sure about which line after which breath, include that line.",
            "If no bad events are present, respond with: 'No significant breath-related issues detected"
        ],
        "events": events
    }

    resp = client.responses.create(
        model=model,
        input=f"Here is structured event data (JSON). Write feedback:\n{json.dumps(prompt)}"
    )
    return resp.output_text


# ----------------------------
# Main high-level function (what your other file can call)
# ----------------------------
def run_breath_alignment_and_feedback(
    mp4_path: str,
    breath_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    out_dir: str,
    whisper_device: str = "cpu",
    whisper_model_size: str = "small",
    do_llm: bool = False,
    llm_model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    This is the function you want to call from elsewhere, passing the two dicts.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) extract audio
    audio_path = os.path.join(out_dir, "_extracted.wav")
    extract_audio_ffmpeg(mp4_path, audio_path, sr=16000)

    # 2) breaths from dict
    breaths = breaths_from_pose_dict(breath_dict)

    # 3) transcript words
    words = transcribe_words_whisperx(audio_path, device=whisper_device, model_size=whisper_model_size)

    # 4) attach words to breath events
    breaths = attach_words_to_breaths(breaths, words, max_gap_s=2.0)

    # 5) assign bad regions -> breaths
    assignments = assign_bad_to_breaths(breaths, bad_dict, pre_window_s=10.0, post_window_s=3.0)

    # 6) build alignment payload + save
    alignment = build_alignment_payload(mp4_path, audio_path, breaths, words, assignments)
    align_path = os.path.join(out_dir, "breath_bad_alignment.json")
    with open(align_path, "w") as f:
        json.dump(alignment, f, indent=2)

    # 7) optional LLM feedback
    feedback_text = None
    if do_llm:
        feedback_text = llm_generate_feedback_openai(alignment, model=llm_model)
        with open(os.path.join(out_dir, "feedback.txt"), "w") as f:
            f.write(feedback_text.strip() + "\n")
        alignment["feedback_txt"] = os.path.join(out_dir, "feedback.txt")

    alignment["alignment_json"] = align_path
    return alignment


# ----------------------------
# CLI (for testing / running standalone)
# ----------------------------
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True, help="Path to input mp4/video")
    ap.add_argument("--breaths_json", required=True, help="Path to breaths dict JSON (contains inhalations)")
    ap.add_argument("--bad_json", required=True, help="Path to bad dict JSON (strain/collapse regions)")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--whisper_device", default="cpu", choices=["cpu", "cuda"], help="WhisperX device")
    ap.add_argument("--whisper_model", default="small", help="Whisper model size: tiny/base/small/medium/large-v2 etc.")
    ap.add_argument("--do_llm", action="store_true", help="Call OpenAI to generate feedback.txt")
    ap.add_argument("--llm_model", default="gpt-5.2", help="OpenAI model name for feedback")
    args = ap.parse_args()

    breath_dict = _load_json(args.breaths_json)
    bad_dict = _load_json(args.bad_json)

    alignment = run_breath_alignment_and_feedback(
        mp4_path=args.mp4,
        breath_dict=breath_dict,
        bad_dict=bad_dict,
        out_dir=args.out_dir,
        whisper_device=args.whisper_device,
        whisper_model_size=args.whisper_model,
        do_llm=args.do_llm,
        llm_model=args.llm_model,
    )

    print("Wrote:", alignment["alignment_json"])
    if args.do_llm and alignment.get("feedback_txt"):
        print("Wrote:", alignment["feedback_txt"])


if __name__ == "__main__":
    main()
