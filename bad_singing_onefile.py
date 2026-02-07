#!/usr/bin/env python3
"""
Bad Singing Detection (ONE FILE) — Strain + Collapse ONLY (no ML)

Filtered output:
- results.json contains ONLY top-k strain + top-k collapse
- keeps ONLY regions lasting >= min_dur_keep_s
- NEW FIX: drops any region that occurs in the first/last `edge_ignore_s` seconds
  (prevents the “last second” phantom datapoint)

Run:
  pip install numpy scipy librosa soundfile matplotlib
  python bad_singing_onefile_filtered.py path/to/audio.mp3 --out_dir out
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import csv


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    sr: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    # segmentation (energy-based)
    top_db_split: float = 30.0
    min_phrase_s: float = 0.30
    merge_gap_s: float = 0.20

    # region thresholds (heuristics)
    strain_thr: float = 0.55
    collapse_thr: float = 0.55

    # HF noise ratio band
    hf_low_hz: float = 2000.0
    hf_high_hz: float = 8000.0

    # smoothing
    smooth_s: float = 0.30

    # Output filtering
    min_dur_keep_s: float = 0.50
    top_k_strain: int = 3
    top_k_collapse: int = 3

    # NEW: ignore first/last seconds (fixes end-of-clip phantom events)
    edge_ignore_s: float = 1.0

    # artifacts
    save_timeline: bool = True
    save_plot: bool = True


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sec_to_samples(t: float, sr: int) -> int:
    return int(round(t * sr))

def samples_to_sec(n: int, sr: int) -> float:
    return float(n) / float(sr)

def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")

def duration_s(r: Dict[str, Any]) -> float:
    return float(r["end_s"]) - float(r["start_s"])


# ----------------------------
# Audio load / preprocess
# ----------------------------

def load_audio_mono(path: str, sr: int) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio.")
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y.astype(np.float32), sr

def light_bandpass(y: np.ndarray, sr: int, lo_hz: float = 80.0, hi_hz: float = 8000.0) -> np.ndarray:
    nyq = 0.5 * sr
    lo = max(1.0, lo_hz) / nyq
    hi = min(hi_hz, nyq - 1.0) / nyq
    if hi <= lo:
        return y
    b, a = scipy.signal.butter(4, [lo, hi], btype="band")
    return scipy.signal.lfilter(b, a, y).astype(np.float32)


# ----------------------------
# Segmentation (VAD-ish)
# ----------------------------

def split_phrases_energy(y: np.ndarray, sr: int, cfg: Config) -> List[Tuple[int, int]]:
    intervals = librosa.effects.split(y, top_db=cfg.top_db_split)
    if len(intervals) == 0:
        return []

    merged: List[Tuple[int, int]] = []
    gap_samp = sec_to_samples(cfg.merge_gap_s, sr)

    for (a, b) in intervals:
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a - pb <= gap_samp:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))

    min_samp = sec_to_samples(cfg.min_phrase_s, sr)
    merged = [(a, b) for (a, b) in merged if (b - a) >= min_samp]
    return merged


# ----------------------------
# Feature extraction
# ----------------------------

def framing_params(cfg: Config, sr: int) -> Tuple[int, int]:
    frame_len = int(round(cfg.frame_ms * 1e-3 * sr))
    hop_len = int(round(cfg.hop_ms * 1e-3 * sr))
    frame_len = max(64, frame_len)
    hop_len = max(16, hop_len)
    return frame_len, hop_len

def rms_envelope(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=True)[0]
    return rms.astype(np.float64)

def stft_mag(y: np.ndarray, n_fft: int, hop_len: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=n_fft, center=True))
    return S.astype(np.float64)

def spectral_tilt_per_frame(S: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float64)
    freqs = np.maximum(freqs, 1.0)
    logf = np.log(freqs)

    mag = np.maximum(S, 1e-10)
    logm = np.log(mag)

    x = logf[:, None]
    x_mu = np.mean(x)
    x0 = x - x_mu
    denom = np.sum(x0 * x0) + 1e-12

    y_mu = np.mean(logm, axis=0, keepdims=True)
    y0 = logm - y_mu
    numer = np.sum(x0 * y0, axis=0)
    slope = numer / denom
    return slope.astype(np.float64)

def hf_noise_ratio(S: np.ndarray, sr: int, n_fft: int, low_hz: float, high_hz: float) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float64)
    band = (freqs >= low_hz) & (freqs <= min(high_hz, sr / 2.0))
    band_energy = np.sum(S[band, :] ** 2, axis=0)
    total_energy = np.sum(S ** 2, axis=0) + 1e-12
    return (band_energy / total_energy).astype(np.float64)

def hnr_proxy(y: np.ndarray, sr: int, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = 1 + (len(y) - frame_len) // hop_len if len(y) >= frame_len else 0
    if n_frames <= 0:
        return np.array([], dtype=np.float64)

    hnr = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a = i * hop_len
        frame = y[a:a + frame_len]
        if frame.shape[0] < frame_len:
            break
        frame = frame - np.mean(frame)
        denom = np.sum(frame * frame) + 1e-12
        if denom <= 0:
            hnr[i] = np.nan
            continue

        ac = scipy.signal.correlate(frame, frame, mode="full")
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-12)

        min_lag = int(sr / 600.0)
        max_lag = int(sr / 50.0)
        max_lag = min(max_lag, ac.size - 1)
        if max_lag <= min_lag + 2:
            hnr[i] = np.nan
            continue

        peak = np.max(ac[min_lag:max_lag])
        hnr[i] = float(np.clip(peak, 0.0, 1.0))
    return hnr


# ----------------------------
# Scoring (strain + collapse only)
# ----------------------------

def score_strain_collapse(hf_ratio: np.ndarray, tilt: np.ndarray, hnr: np.ndarray, rms: np.ndarray) -> Dict[str, np.ndarray]:
    def squash(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -6.0, 6.0)
        return 1.0 / (1.0 + np.exp(-z))

    hf_z = zscore_safe(hf_ratio)
    tilt_z = zscore_safe(-tilt)
    hnr_z = zscore_safe(1.0 - hnr)
    strain_mix = 0.50 * hf_z + 0.25 * tilt_z + 0.25 * hnr_z
    strain_score = squash(strain_mix)

    lrms = np.log(np.maximum(rms, 1e-8))
    dl = np.diff(lrms)
    dl = np.concatenate([[0.0], dl])
    drop = np.maximum(0.0, -dl)
    drop_z = zscore_safe(drop)
    collapse_score = squash(drop_z)

    clarity = 1.0 - clamp01(strain_score)
    return {"strain": clamp01(strain_score), "collapse": clamp01(collapse_score), "clarity": clamp01(clarity)}


def pick_regions(score: np.ndarray, times: np.ndarray, thr: float, min_len_s: float = 0.25, merge_gap_s: float = 0.15) -> List[Tuple[float, float, float]]:
    above = score > thr
    if not np.any(above):
        return []

    idx = np.where(above)[0]
    groups = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))

    regions = []
    for a, b in groups:
        s = float(times[a])
        e = float(times[b])
        peak = float(np.max(score[a:b+1]))
        regions.append((s, e, peak))

    merged = []
    for s, e, p in regions:
        if not merged:
            merged.append([s, e, p])
        else:
            ps, pe, pp = merged[-1]
            if s - pe <= merge_gap_s:
                merged[-1][1] = max(pe, e)
                merged[-1][2] = max(pp, p)
            else:
                merged.append([s, e, p])

    out = []
    for s, e, p in merged:
        if (e - s) >= min_len_s:
            out.append((float(s), float(e), float(p)))
    return out


def coaching_note(label: str) -> str:
    if label == "strain":
        return "Tone becomes harsher/noisier here. Try reducing intensity, relaxing tension, and adjusting vowel shaping."
    if label == "collapse":
        return "Energy drops sharply here (phrase collapse). Consider shorter phrasing or resetting support earlier."
    return "Control likely drops here."


# ----------------------------
# NEW: Edge filtering (fix)
# ----------------------------

def drop_edge_regions(regions: List[Dict[str, Any]], audio_dur_s: float, edge_ignore_s: float) -> List[Dict[str, Any]]:
    """
    Drop any region that touches:
      [0, edge_ignore_s) or (audio_dur_s-edge_ignore_s, audio_dur_s]
    This avoids end-of-clip STFT padding artifacts.
    """
    if edge_ignore_s <= 0:
        return regions

    lo = float(edge_ignore_s)
    hi = float(max(0.0, audio_dur_s - edge_ignore_s))

    kept = []
    for r in regions:
        s = float(r["start_s"])
        e = float(r["end_s"])
        if s < lo:
            continue
        if e > hi:
            continue
        kept.append(r)
    return kept


# ----------------------------
# Output filtering
# ----------------------------

def filter_top_regions(regions: List[Dict[str, Any]], label: str, top_k: int, min_dur_s: float) -> List[Dict[str, Any]]:
    cand = [r for r in regions if r.get("label") == label and duration_s(r) >= min_dur_s]
    cand.sort(key=lambda r: (-float(r["peak"]), duration_s(r), float(r["start_s"])))
    return cand[:top_k]


def minimal_results_json(audio_path: str, out_dir: str, cfg: Config,
                         phrases: List[Tuple[int,int]], scores: Dict[str, float],
                         top_strain: List[Dict[str, Any]], top_collapse: List[Dict[str, Any]]) -> Dict[str, Any]:
    def pack(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": r["label"],
            "start_s": round(float(r["start_s"]), 3),
            "end_s": round(float(r["end_s"]), 3),
            "duration_s": round(duration_s(r), 3),
            "peak": round(float(r["peak"]), 3),
            "note": coaching_note(r["label"]),
        }

    return {
        "input_audio": os.path.abspath(audio_path),
        "sample_rate": cfg.sr,
        "filters": {
            "min_duration_s": cfg.min_dur_keep_s,
            "top_k_strain": cfg.top_k_strain,
            "top_k_collapse": cfg.top_k_collapse,
            "edge_ignore_s": cfg.edge_ignore_s,
        },
        "scores": {k: round(float(v), 3) for k, v in scores.items()},
        "mistakes": {
            "strain": [pack(r) for r in top_strain],
            "collapse": [pack(r) for r in top_collapse],
        },
        "artifacts": {
            "timeline_csv": os.path.abspath(os.path.join(out_dir, "timeline.csv")) if cfg.save_timeline else None,
            "report_png": os.path.abspath(os.path.join(out_dir, "report.png")) if cfg.save_plot else None,
        },
        "disclaimer": "Coaching-oriented acoustic pattern detection, not medical diagnosis."
    }


# ----------------------------
# Plot + timeline
# ----------------------------

def save_report_plot(path: str, y: np.ndarray, sr: int, times: np.ndarray,
                     strain_s: np.ndarray, collapse_s: np.ndarray,
                     regions_to_highlight: List[Dict[str, Any]]) -> None:
    t_audio = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_audio, y)
    ax1.set_title("Waveform")
    ax1.set_xlim(0, t_audio[-1])

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(times, strain_s)
    ax2.set_ylim(0, 1)
    ax2.set_title("Strain / Harshness (smoothed)")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(times, collapse_s)
    ax3.set_ylim(0, 1)
    ax3.set_title("Phrase Collapse (smoothed)")
    ax3.set_xlabel("Time (s)")

    for r in regions_to_highlight:
        s = float(r["start_s"])
        e = float(r["end_s"])
        for ax in (ax2, ax3):
            ax.axvspan(s, e, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_timeline_csv(out_path: str, times: np.ndarray, rms: np.ndarray, hnr: np.ndarray,
                       hf_ratio: np.ndarray, tilt: np.ndarray, heads: Dict[str, np.ndarray]) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "rms", "hnr_proxy", "hf_ratio", "tilt", "strain", "collapse", "clarity"])
        for i in range(len(times)):
            w.writerow([
                float(times[i]),
                float(rms[i]) if np.isfinite(rms[i]) else "",
                float(hnr[i]) if np.isfinite(hnr[i]) else "",
                float(hf_ratio[i]) if np.isfinite(hf_ratio[i]) else "",
                float(tilt[i]) if np.isfinite(tilt[i]) else "",
                float(heads["strain"][i]),
                float(heads["collapse"][i]),
                float(heads["clarity"][i]),
            ])


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline(audio_path: str, out_dir: str, cfg: Config, separate_vocals: bool = False) -> Dict[str, Any]:
    ensure_dir(out_dir)

    y, sr = load_audio_mono(audio_path, cfg.sr)
    y = light_bandpass(y, sr)

    if separate_vocals:
        y = demucs_separate_to_vocals(y, sr, out_dir)

    audio_dur_s = float(len(y) / sr)

    phrases = split_phrases_energy(y, sr, cfg)
    if not phrases:
        raise RuntimeError("No voiced phrases detected. Try lowering --top_db_split or use cleaner vocal audio.")

    frame_len, hop_len = framing_params(cfg, sr)
    n_fft = 1024 if sr <= 22050 else 2048

    rms = rms_envelope(y, frame_len, hop_len)
    S = stft_mag(y, n_fft=n_fft, hop_len=hop_len)
    tilt = spectral_tilt_per_frame(S, sr, n_fft=n_fft)
    hf_ratio = hf_noise_ratio(S, sr, n_fft=n_fft, low_hz=cfg.hf_low_hz, high_hz=cfg.hf_high_hz)
    hnr = hnr_proxy(y, sr, frame_len, hop_len)

    L = min(len(rms), S.shape[1], len(tilt), len(hf_ratio), len(hnr))
    rms = rms[:L]
    tilt = tilt[:L]
    hf_ratio = hf_ratio[:L]
    hnr = hnr[:L]
    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hop_len)

    heads = score_strain_collapse(hf_ratio, tilt, hnr, rms)

    smooth_win = int(round(cfg.smooth_s / (cfg.hop_ms * 1e-3)))
    strain_s = smooth(heads["strain"], smooth_win)
    collapse_s = smooth(heads["collapse"], smooth_win)

    strain_regions = pick_regions(strain_s, times, cfg.strain_thr)
    collapse_regions = pick_regions(collapse_s, times, cfg.collapse_thr)

    regions: List[Dict[str, Any]] = []
    for (s, e, p) in strain_regions:
        regions.append({"start_s": float(s), "end_s": float(e), "peak": float(p), "label": "strain"})
    for (s, e, p) in collapse_regions:
        regions.append({"start_s": float(s), "end_s": float(e), "peak": float(p), "label": "collapse"})

    # NEW: remove edge artifacts (start/end)
    regions = drop_edge_regions(regions, audio_dur_s=audio_dur_s, edge_ignore_s=cfg.edge_ignore_s)

    regions.sort(key=lambda r: (r["label"], -r["peak"], r["start_s"]))

    # Aggregate scores
    strain_overall = float(np.nanmean(strain_s))
    collapse_overall = float(np.nanmean(collapse_s))
    clarity_overall = float(np.nanmean(heads["clarity"]))
    overall = float(np.clip(0.60 * strain_overall + 0.40 * collapse_overall, 0.0, 1.0))
    scores = {"overall_problem_score": overall, "strain": strain_overall, "collapse": collapse_overall, "clarity": clarity_overall}

    # Filter for minimal JSON output
    top_strain = filter_top_regions(regions, "strain", cfg.top_k_strain, cfg.min_dur_keep_s)
    top_collapse = filter_top_regions(regions, "collapse", cfg.top_k_collapse, cfg.min_dur_keep_s)

    # Save artifacts (optional)
    if cfg.save_timeline:
        write_timeline_csv(os.path.join(out_dir, "timeline.csv"), times, rms, hnr, hf_ratio, tilt, heads)

    if cfg.save_plot:
        highlight = sorted(top_strain + top_collapse, key=lambda r: r["start_s"])
        save_report_plot(os.path.join(out_dir, "report.png"), y, sr, times, strain_s, collapse_s, highlight)

    # Minimal results.json
    minimal = minimal_results_json(audio_path, out_dir, cfg, phrases, scores, top_strain, top_collapse)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(minimal, f, indent=2)

    return minimal


# ----------------------------
# Optional: Demucs vocal separation
# ----------------------------

def demucs_separate_to_vocals(y: np.ndarray, sr: int, out_dir: str) -> np.ndarray:
    import subprocess
    tmp_in = os.path.join(out_dir, "_tmp_input.wav")
    sf.write(tmp_in, y, sr)

    demucs_out = os.path.join(out_dir, "_demucs")
    ensure_dir(demucs_out)

    cmd = ["python", "-m", "demucs", "-o", demucs_out, tmp_in]
    subprocess.check_call(cmd)

    vocals_path = None
    for root, _, files in os.walk(demucs_out):
        for fn in files:
            if fn.lower() == "vocals.wav":
                vocals_path = os.path.join(root, fn)
                break
        if vocals_path:
            break
    if vocals_path is None:
        raise RuntimeError("Could not find vocals.wav in demucs output.")

    v, _ = librosa.load(vocals_path, sr=sr, mono=True)
    peak = np.max(np.abs(v)) + 1e-9
    v = v / peak
    return v.astype(np.float32)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Path to audio file (.wav/.mp3/.m4a)")
    p.add_argument("--out_dir", default="bad_singing_out", help="Output directory")
    p.add_argument("--sr", type=int, default=16000)

    p.add_argument("--top_db_split", type=float, default=30.0)
    p.add_argument("--min_phrase_s", type=float, default=0.30)
    p.add_argument("--merge_gap_s", type=float, default=0.20)
    p.add_argument("--strain_thr", type=float, default=0.55)
    p.add_argument("--collapse_thr", type=float, default=0.55)

    p.add_argument("--min_dur_keep_s", type=float, default=0.50)
    p.add_argument("--top_k_strain", type=int, default=3)
    p.add_argument("--top_k_collapse", type=int, default=3)

    # NEW: edge ignore
    p.add_argument("--edge_ignore_s", type=float, default=1.0, help="Ignore events in first/last N seconds")

    p.add_argument("--no_timeline", action="store_true", help="Do not write timeline.csv")
    p.add_argument("--no_plot", action="store_true", help="Do not write report.png")
    p.add_argument("--separate_vocals", action="store_true", help="Run Demucs vocal separation first (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        sr=args.sr,
        top_db_split=args.top_db_split,
        min_phrase_s=args.min_phrase_s,
        merge_gap_s=args.merge_gap_s,
        strain_thr=args.strain_thr,
        collapse_thr=args.collapse_thr,
        min_dur_keep_s=args.min_dur_keep_s,
        top_k_strain=args.top_k_strain,
        top_k_collapse=args.top_k_collapse,
        edge_ignore_s=args.edge_ignore_s,
        save_timeline=(not args.no_timeline),
        save_plot=(not args.no_plot),
    )

    result = run_pipeline(args.audio, args.out_dir, cfg, separate_vocals=args.separate_vocals)

    print("\n=== Filtered Mistakes Written to results.json ===")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")
    print("\nStrain (top):")
    for r in result["mistakes"]["strain"]:
        print(f'  - {r["start_s"]:.2f}s–{r["end_s"]:.2f}s  dur={r["duration_s"]:.2f}s  peak={r["peak"]:.2f}')
    print("\nCollapse (top):")
    for r in result["mistakes"]["collapse"]:
        print(f'  - {r["start_s"]:.2f}s–{r["end_s"]:.2f}s  dur={r["duration_s"]:.2f}s  peak={r["peak"]:.2f}')

    print("\nDone.\n")


if __name__ == "__main__":
    main()
