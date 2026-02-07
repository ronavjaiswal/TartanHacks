#!/usr/bin/env python3
"""
breath_coach_onefile.py

Single-file breath coaching integrator.

Inputs:
  - mp4/audio path
  - breaths_dict: your pose/audio fusion output (uses inhalations[].t_start as breath start)
  - bad_dict: detector output dict (mistakes.strain/collapse regions)

Outputs (returned dict):
  - extracted audio path used
  - words with timestamps (if transcription succeeds)
  - bad regions
  - breath events (start only)
  - assignments: bad region -> chosen breath (+ word anchors)
  - coaching_lines (deterministic)
  - llm_coaching (optional, if --llm and OPENAI_API_KEY is set)

Env vars:
  - OPENAI_API_KEY: set to your key to enable LLM calls
    (if missing, treated as "NotImplemented")

CLI:
  python breath_coach_onefile.py --input test.mp4 --bad_json bad.json --breaths_json breaths.json --out out --llm

Transcription:
  Preferred: whisperx (word-level alignment)
    pip install -U whisperx
  Fallback: openai-whisper (segment timestamps; we approximate word timestamps if needed)
    pip install -U openai-whisper

ffmpeg:
  mac: brew install ffmpeg
  ubuntu: sudo apt-get install ffmpeg
  windows: install ffmpeg and add to PATH
"""

from __future__ import annotations

import os
import re
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ------------------------- Helpers -------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _is_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

def _run_ffmpeg_extract_audio(input_path: str, out_wav_path: str, sr: int = 16000) -> None:
    """
    Extract mono 16k wav for transcription. Overwrites out_wav_path.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        out_wav_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it's on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}")

def _pick_audio_path(input_path: str, out_dir: str, sr: int = 16000) -> str:
    """
    If input is video, extract audio to out_dir/_extracted.wav.
    If input is audio, return as-is (still may be re-encoded by ffmpeg if desired; here we keep it).
    """
    _ensure_dir(out_dir)
    if _is_video(input_path):
        out_wav = os.path.join(out_dir, "_extracted.wav")
        _run_ffmpeg_extract_audio(input_path, out_wav, sr=sr)
        return out_wav
    else:
        # Optionally you could still normalize/convert; keeping simple.
        return input_path

def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _sort_by_key(items: List[dict], key: str) -> List[dict]:
    return sorted(items, key=lambda d: _safe_float(d.get(key, float("inf"))))

# ------------------------- Breath + Mistakes -------------------------

@dataclass
class BreathEvent:
    start: float
    confidence: float = 1.0
    source: str = "inhalations"

@dataclass
class BadRegion:
    label: str               # "strain" | "collapse"
    start_s: float
    end_s: float
    peak: float
    note: str

def extract_breath_events(breaths_dict: Dict[str, Any]) -> List[BreathEvent]:
    """
    Your note: durations are unreliable, starts are reliable.
    We treat each inhalations[].t_start as a breath event time.
    """
    inh = breaths_dict.get("inhalations", []) or []
    events: List[BreathEvent] = []
    for it in inh:
        t = _safe_float(it.get("t_start"))
        if math.isfinite(t):
            events.append(BreathEvent(start=t, confidence=1.0, source="inhalations"))
    events.sort(key=lambda b: b.start)
    return events

def extract_bad_regions(bad_dict: Dict[str, Any]) -> List[BadRegion]:
    mistakes = (bad_dict.get("mistakes") or {})
    out: List[BadRegion] = []
    for label in ("strain", "collapse"):
        for r in (mistakes.get(label) or []):
            out.append(
                BadRegion(
                    label=label,
                    start_s=_safe_float(r.get("start_s")),
                    end_s=_safe_float(r.get("end_s")),
                    peak=_safe_float(r.get("peak")),
                    note=str(r.get("note") or ""),
                )
            )
    out.sort(key=lambda br: br.start_s)
    return out

# ------------------------- Assignment: bad region -> breath -------------------------

def _nearest_breath_before(breaths: List[BreathEvent], t: float) -> Optional[BreathEvent]:
    before = [b for b in breaths if b.start <= t]
    return max(before, key=lambda b: b.start) if before else None

def _nearest_breath_after(breaths: List[BreathEvent], t: float) -> Optional[BreathEvent]:
    after = [b for b in breaths if b.start > t]
    return min(after, key=lambda b: b.start) if after else None

def assign_regions_to_breaths(
    regions: List[BadRegion],
    breaths: List[BreathEvent],
    pre_window_s: float = 10.0,
    post_window_s: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    For each bad region, choose a breath:
      - Prefer closest breath BEFORE region start within pre_window_s
      - Else closest breath AFTER within post_window_s
      - Else whatever exists (before/after), or None
    """
    out: List[Dict[str, Any]] = []
    for r in regions:
        prev_b = _nearest_breath_before(breaths, r.start_s)
        next_b = _nearest_breath_after(breaths, r.start_s)

        chosen: Optional[BreathEvent] = None
        relation: Optional[str] = None

        if prev_b and (r.start_s - prev_b.start) <= pre_window_s:
            chosen = prev_b
            relation = "before"
        elif next_b and (next_b.start - r.start_s) <= post_window_s:
            chosen = next_b
            relation = "after"
        else:
            chosen = prev_b or next_b
            relation = "before" if chosen is prev_b else ("after" if chosen is next_b else None)

        out.append({
            "label": r.label,
            "region": {
                "start_s": r.start_s,
                "end_s": r.end_s,
                "peak": r.peak,
                "note": r.note,
            },
            "breath": (None if chosen is None else {
                "start": chosen.start,
                "confidence": chosen.confidence,
                "source": chosen.source,
            }),
            "breath_relation": relation
        })
    return out

# ------------------------- Transcription (word timestamps) -------------------------

def transcribe_words(audio_path: str, device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Returns list of words: [{"word": str, "start": float, "end": float}, ...]

    Preferred: whisperx for aligned word timestamps.
    Fallback: openai-whisper segments (no word timestamps) -> we approximate word times by
              distributing words uniformly across the segment.

    You can install:
      pip install -U whisperx
    or:
      pip install -U openai-whisper
    """
    # Try WhisperX
    try:
        import whisperx  # type: ignore

        model = whisperx.load_model("small", device=device)
        result = model.transcribe(audio_path)

        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device=device)

        words: List[Dict[str, Any]] = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []) or []:
                if w.get("start") is None or w.get("end") is None:
                    continue
                token = str(w.get("word") or "").strip()
                if token:
                    words.append({"word": token, "start": float(w["start"]), "end": float(w["end"])})
        return words

    except ImportError:
        pass
    except Exception:
        # If whisperx exists but fails, try fallback
        pass

    # Fallback: openai-whisper
    try:
        import whisper  # type: ignore
    except ImportError:
        raise RuntimeError(
            "No transcription backend available. Install one:\n"
            "  pip install -U whisperx\n"
            "or\n"
            "  pip install -U openai-whisper"
        )

    model = whisper.load_model("small")
    res = model.transcribe(audio_path, fp16=False)
    segments = res.get("segments", []) or []

    words_out: List[Dict[str, Any]] = []
    for seg in segments:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", s0))
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        # crude word splitting
        toks = [t for t in re.split(r"\s+", text) if t]
        if not toks or s1 <= s0:
            continue
        dur = s1 - s0
        step = dur / max(len(toks), 1)
        for i, tok in enumerate(toks):
            w0 = s0 + i * step
            w1 = min(s1, w0 + step)
            words_out.append({"word": tok, "start": w0, "end": w1})
    return words_out

def _nearest_word(words: List[Dict[str, Any]], t: float, direction: str = "before") -> Optional[Dict[str, Any]]:
    if not words:
        return None
    if direction == "before":
        cand = [w for w in words if float(w["start"]) <= t]
        return max(cand, key=lambda w: float(w["start"])) if cand else None
    else:
        cand = [w for w in words if float(w["start"]) > t]
        return min(cand, key=lambda w: float(w["start"])) if cand else None

def attach_word_anchors(assignments: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each assignment (region -> breath), attach nearest word before/after the breath start.
    """
    for a in assignments:
        breath = a.get("breath")
        if not breath:
            a["word_anchor"] = None
            continue
        bt = float(breath["start"])
        w_before = _nearest_word(words, bt, "before")
        w_after = _nearest_word(words, bt, "after")
        a["word_anchor"] = {
            "before": w_before,
            "after": w_after
        }
    return assignments

# ------------------------- Deterministic coaching lines -------------------------

def make_coaching_lines(assignments: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for a in assignments:
        label = a["label"]
        r = a["region"]
        rs, re_, peak = float(r["start_s"]), float(r["end_s"]), float(r["peak"])
        breath = a.get("breath")
        rel = a.get("breath_relation")

        wa = a.get("word_anchor") or {}
        w_before = (wa.get("before") or {}).get("word")
        w_before_t = (wa.get("before") or {}).get("start")
        w_after = (wa.get("after") or {}).get("word")
        w_after_t = (wa.get("after") or {}).get("start")

        if breath:
            bt = float(breath["start"])
            if w_before:
                anchor = f"near '{w_before}' ({float(w_before_t):.2f}s)"
            elif w_after:
                anchor = f"near '{w_after}' ({float(w_after_t):.2f}s)"
            else:
                anchor = "near (no word anchor)"

            if label == "collapse":
                lines.append(
                    f"[COLLAPSE] {rs:.2f}-{re_:.2f}s peak={peak:.2f}. "
                    f"Closest breath {rel} at {bt:.2f}s {anchor}. "
                    f"Suggestion: plan a reset earlier (shorter phrase / breathe before the phrase end)."
                )
            else:
                lines.append(
                    f"[STRAIN] {rs:.2f}-{re_:.2f}s peak={peak:.2f}. "
                    f"Closest breath {rel} at {bt:.2f}s {anchor}. "
                    f"Suggestion: lighten onset, reduce push, and consider breathing sooner before this section."
                )
        else:
            if label == "collapse":
                lines.append(
                    f"[COLLAPSE] {rs:.2f}-{re_:.2f}s peak={peak:.2f}. "
                    f"No nearby breath detected. Suggest: earlier reset/breath before this phrase."
                )
            else:
                lines.append(
                    f"[STRAIN] {rs:.2f}-{re_:.2f}s peak={peak:.2f}. "
                    f"No nearby breath detected. Suggest: reduce intensity and check support/relaxation."
                )
    return lines

# ------------------------- Optional OpenAI LLM polishing -------------------------

def llm_polish_feedback(
    transcript_words: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    deterministic_lines: List[str],
    model: str = "gpt-5.2",
) -> str:
    """
    Uses OpenAI Responses API to turn structured findings into natural coaching.

    Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    api_key = os.getenv("OPENAI_API_KEY", "NotImplemented")
    if not api_key or api_key == "NotImplemented":
        raise RuntimeError("OPENAI_API_KEY not set. Set it or run without --llm.")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

    # Build a compact transcript preview (avoid dumping thousands of words)
    # We include the first ~250 and last ~250 words with timestamps.
    def _pack_words(words: List[Dict[str, Any]], max_words: int = 250) -> str:
        if not words:
            return ""
        head = words[:max_words]
        tail = words[-max_words:] if len(words) > max_words else []
        def fmt(ws):
            return " ".join([f"{w['word']}@{float(w['start']):.2f}" for w in ws])
        if tail:
            return fmt(head) + " ... " + fmt(tail)
        return fmt(head)

    transcript_preview = _pack_words(transcript_words, 200)

    payload = {
        "transcript_preview": transcript_preview,
        "assignments": assignments,
        "deterministic_lines": deterministic_lines,
        "instructions": (
            "You are a vocal coach. Produce 3–8 bullet points max.\n"
            "Each bullet should mention:\n"
            " - the problematic region time range\n"
            " - the nearest breath time (and word if present)\n"
            " - a concrete actionable suggestion: where to breathe, how to adjust intensity/support\n"
            "Avoid medical claims. Keep it practical.\n"
            "If you recognise the song, and knwo where the breath shoudl exactly be take, mention the line.\n"
        ),
    }

    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return only the bullet list. No preamble."},
            {"role": "user", "content": json.dumps(payload)}
        ],
    )
    # SDK provides output_text convenience in docs; fallback if missing
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # generic extraction
    try:
        return resp.output[0].content[0].text  # type: ignore
    except Exception:
        return str(resp)

# ------------------------- Main public function -------------------------

def analyze_breaths_and_bad_regions(
    input_media_path: str,
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    out_dir: str = "out_breathcoach",
    do_transcript: bool = True,
    transcript_device: str = "cpu",
    pre_window_s: float = 10.0,
    post_window_s: float = 3.0,
    use_llm: bool = False,
    llm_model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Importable function.

    Returns dict with:
      - audio_used
      - breaths (start times)
      - bad_regions
      - words (optional)
      - assignments (+ word anchors if words present)
      - coaching_lines
      - llm_coaching (optional)
    """
    _ensure_dir(out_dir)
    audio_used = _pick_audio_path(input_media_path, out_dir, sr=int(bad_dict.get("sample_rate", 16000) or 16000))

    breaths = extract_breath_events(breaths_dict)
    regions = extract_bad_regions(bad_dict)

    assignments = assign_regions_to_breaths(regions, breaths, pre_window_s=pre_window_s, post_window_s=post_window_s)

    words: List[Dict[str, Any]] = []
    if do_transcript:
        words = transcribe_words(audio_used, device=transcript_device)
        assignments = attach_word_anchors(assignments, words)

    coaching_lines = make_coaching_lines(assignments)

    result: Dict[str, Any] = {
        "input": input_media_path,
        "audio_used": audio_used,
        "breaths": [{"start": b.start, "confidence": b.confidence, "source": b.source} for b in breaths],
        "bad_regions": [{"label": r.label, "start_s": r.start_s, "end_s": r.end_s, "peak": r.peak, "note": r.note} for r in regions],
        "words": words if do_transcript else None,
        "assignments": assignments,
        "coaching_lines": coaching_lines,
        "llm_coaching": None,
    }

    if use_llm:
        try:
            result["llm_coaching"] = llm_polish_feedback(words, assignments, coaching_lines, model=llm_model)
        except Exception as e:
            # Don't crash; return the deterministic output and surface the error
            result["llm_coaching"] = f"[LLM disabled/error] {e}"

    # also persist a json artifact for debugging
    with open(os.path.join(out_dir, "breath_coach_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

# ------------------------- CLI -------------------------

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mp4/audio input")
    ap.add_argument("--breaths_json", required=True, help="json file containing breaths_dict format")
    ap.add_argument("--bad_json", required=True, help="json file containing bad_dict format")
    ap.add_argument("--out", default="out_breathcoach")
    ap.add_argument("--no_transcript", action="store_true")
    ap.add_argument("--device", default="cpu", help="transcription device: cpu/cuda")
    ap.add_argument("--pre_window_s", type=float, default=10.0)
    ap.add_argument("--post_window_s", type=float, default=3.0)
    ap.add_argument("--llm", action="store_true", help="Use OpenAI to polish coaching (requires OPENAI_API_KEY)")
    ap.add_argument("--llm_model", default="gpt-5.2")
    args = ap.parse_args()

    breaths_dict = _load_json(args.breaths_json)
    bad_dict = _load_json(args.bad_json)

    res = analyze_breaths_and_bad_regions(
        input_media_path=args.input,
        breaths_dict=breaths_dict,
        bad_dict=bad_dict,
        out_dir=args.out,
        do_transcript=not args.no_transcript,
        transcript_device=args.device,
        pre_window_s=args.pre_window_s,
        post_window_s=args.post_window_s,
        use_llm=args.llm,
        llm_model=args.llm_model,
    )

    print("\nDeterministic coaching lines:")
    for line in res["coaching_lines"]:
        print(" -", line)

    if args.llm:
        print("\nLLM coaching:")
        print(res["llm_coaching"])

if __name__ == "__main__":
    main()
