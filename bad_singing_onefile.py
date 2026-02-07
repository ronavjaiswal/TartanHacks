#!/usr/bin/env python3
"""
Bad Singing Detection (ONE FILE) — Strain + Collapse ONLY (no ML)

Fixes:
- Removed the duration >= 0.5s filter entirely (top-k only).
- Much less noise via:
  * phrase/VAD gating mask (only score inside voiced/active segments)
  * robust normalization (median/MAD instead of mean/std)
  * improved strain proxy (adds spectral flatness; less fooled by music/transients)
  * collapse uses smoothed log-RMS slope inside phrases

MP4/video input:
- If input is .mp4/.mov/.mkv/... we extract audio via ffmpeg to out_dir/_extracted.wav

Install:
  pip install numpy scipy librosa soundfile matplotlib

ffmpeg (required for mp4/video):
  mac:    brew install ffmpeg
  ubuntu: sudo apt-get install ffmpeg
  windows: install ffmpeg + add to PATH

Run:
  python bad_singing_onefile_fixed.py input.mp3 --out_dir out
  python bad_singing_onefile_fixed.py input.mp4 --out_dir out

Outputs:
  out_dir/results.json   (minimal: only top-k strain + top-k collapse)
  out_dir/timeline.csv   (optional)
  out_dir/report.png     (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
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

    # segmentation (energy-based) -> used to create phrase mask
    top_db_split: float = 30.0
    min_phrase_s: float = 0.30
    merge_gap_s: float = 0.20

    # region thresholds (heuristics)
    strain_thr: float = 0.62
    collapse_thr: float = 0.62

    # HF noise ratio band
    hf_low_hz: float = 2000.0
    hf_high_hz: float = 8000.0

    # smoothing
    smooth_s: float = 0.30

    # keep top mistakes only
    top_k_strain: int = 3
    top_k_collapse: int = 3

    # ignore start/end seconds (prevents end padding artifacts)
    edge_ignore_s: float = 1.0

    # extra gating: only consider frames where RMS is above this percentile within phrases
    # (helps kill quiet-room noise regions)
    phrase_rms_percentile: float = 30.0

    # artifacts
    save_timeline: bool = True
    save_plot: bool = True


# ----------------------------
# Utilities
# ----------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score via median/MAD (less sensitive to spikes)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")

def duration_s(r: Dict[str, Any]) -> float:
    return float(r["end_s"]) - float(r["start_s"])

def squash(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-z))

def sec_to_samples(t: float, sr: int) -> int:
    return int(round(t * sr))


# ----------------------------
# MP4/video -> audio extraction
# ----------------------------

def is_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS

def extract_audio_ffmpeg(video_path: str, out_wav_path: str, sr: int) -> None:
    """
    Extract mono wav at target sr using ffmpeg.
    """
    # ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 out.wav
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), out_wav_path]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it's on PATH.")
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg failed to extract audio. Try converting the video to wav/mp3 first.")


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
# Segmentation (phrase mask)
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

def make_phrase_frame_mask(times: np.ndarray, phrases: List[Tuple[int,int]], sr: int) -> np.ndarray:
    """Frame mask: True if time falls within any phrase interval."""
    mask = np.zeros_like(times, dtype=bool)
    for a, b in phrases:
        s = a / sr
        e = b / sr
        mask |= (times >= s) & (times <= e)
    return mask


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
    return librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=True)[0].astype(np.float64)

def stft_mag(y: np.ndarray, n_fft: int, hop_len: int) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=n_fft, center=True)).astype(np.float64)

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
    return (numer / denom).astype(np.float64)

def hf_noise_ratio(S: np.ndarray, sr: int, n_fft: int, low_hz: float, high_hz: float) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float64)
    band = (freqs >= low_hz) & (freqs <= min(high_hz, sr / 2.0))
    band_energy = np.sum(S[band, :] ** 2, axis=0)
    total_energy = np.sum(S ** 2, axis=0) + 1e-12
    return (band_energy / total_energy).astype(np.float64)

def spectral_flatness(S: np.ndarray) -> np.ndarray:
    """Noise-likeness proxy; higher flatness => more noisy/aperiodic."""
    # librosa.feature.spectral_flatness expects power spectrogram, but mag works OK as proxy
    flat = librosa.feature.spectral_flatness(S=S).squeeze(axis=0)
    return flat.astype(np.float64)

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
# Scoring (robust + gated)
# ----------------------------

def score_strain_collapse(
    hf_ratio: np.ndarray,
    tilt: np.ndarray,
    flat: np.ndarray,
    hnr: np.ndarray,
    rms: np.ndarray,
    phrase_mask: np.ndarray,
    cfg: Config
) -> Dict[str, np.ndarray]:
    """
    strain: robust combo of HF ratio + spectral flatness + low HNR + tilt proxy
    collapse: robust slope/drop in smoothed log-RMS
    Only score inside phrases + RMS-above-percentile gating inside phrases.
    """

    # Base gate: only frames in phrases
    gate = phrase_mask.copy()

    # Extra gate: remove very-quiet frames within phrases (helps kill noise)
    if np.any(gate):
        rms_phrase = rms[gate]
        thr = np.percentile(rms_phrase, cfg.phrase_rms_percentile)
        gate &= (rms >= thr)

    # robust normalized features (only compute distribution on gated frames; fallback to global)
    def rnorm(x: np.ndarray) -> np.ndarray:
        if np.any(gate):
            z = (x - np.nanmedian(x[gate])) / (1.4826 * (np.nanmedian(np.abs(x[gate] - np.nanmedian(x[gate]))) + 1e-8))
            return z
        return robust_z(x)

    hf_z = rnorm(hf_ratio)
    flat_z = rnorm(flat)
    tilt_z = rnorm(-tilt)       # invert so "less-negative tilt" increases score
    hnr_z = rnorm(1.0 - hnr)    # lower hnr -> higher noise/strain

    # Strain mix: lean more on flatness + hf_ratio (better for noisy/pressed)
    strain_mix = 0.35 * hf_z + 0.35 * flat_z + 0.20 * hnr_z + 0.10 * tilt_z
    strain = squash(strain_mix)

    # Collapse: smoothed log-RMS slope (drop)
    lrms = np.log(np.maximum(rms, 1e-8))
    # smooth log rms to reduce jitter
    win = max(3, int(round(cfg.smooth_s / (cfg.hop_ms * 1e-3))))
    lrms_s = smooth(lrms, win)
    dl = np.diff(lrms_s)
    dl = np.concatenate([[0.0], dl])
    drop = np.maximum(0.0, -dl)
    drop_z = rnorm(drop)
    collapse = squash(drop_z)

    # Apply gate (outside: zero)
    strain = np.where(gate, strain, 0.0)
    collapse = np.where(gate, collapse, 0.0)
    clarity = 1.0 - clamp01(strain)

    return {"strain": clamp01(strain), "collapse": clamp01(collapse), "clarity": clamp01(clarity), "gate": gate}


# ----------------------------
# Regions
# ----------------------------

def pick_regions(score: np.ndarray, times: np.ndarray, thr: float, min_len_s: float = 0.20, merge_gap_s: float = 0.15) -> List[Tuple[float, float, float]]:
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

    return [(float(s), float(e), float(p)) for (s, e, p) in merged if (e - s) >= min_len_s]

def drop_edge_regions(regions: List[Dict[str, Any]], audio_dur_s: float, edge_ignore_s: float) -> List[Dict[str, Any]]:
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

def coaching_note(label: str) -> str:
    if label == "strain":
        return "Likely harsh/pressed/noisy phonation. Try reducing intensity, relaxing jaw/neck, and re-shaping vowels."
    if label == "collapse":
        return "Likely support drop / phrase collapse. Try shorter phrases, earlier reset, or lower intensity at phrase end."
    return "Control likely drops here."


def top_k(regions: List[Dict[str, Any]], label: str, k: int) -> List[Dict[str, Any]]:
    cand = [r for r in regions if r.get("label") == label]
    cand.sort(key=lambda r: (-float(r["peak"]), float(r["start_s"])))
    return cand[:k]


# ----------------------------
# Artifacts
# ----------------------------

def save_report_plot(path: str, y: np.ndarray, sr: int, times: np.ndarray,
                     strain_s: np.ndarray, collapse_s: np.ndarray,
                     highlight: List[Dict[str, Any]]) -> None:
    t_audio = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_audio, y)
    ax1.set_title("Waveform")
    ax1.set_xlim(0, t_audio[-1])

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(times, strain_s)
    ax2.set_ylim(0, 1)
    ax2.set_title("Strain (gated + robust)")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(times, collapse_s)
    ax3.set_ylim(0, 1)
    ax3.set_title("Collapse (gated + robust)")
    ax3.set_xlabel("Time (s)")

    for r in highlight:
        s = float(r["start_s"])
        e = float(r["end_s"])
        ax2.axvspan(s, e, alpha=0.25)
        ax3.axvspan(s, e, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def write_timeline_csv(out_path: str, times: np.ndarray, rms: np.ndarray, hnr: np.ndarray,
                       hf_ratio: np.ndarray, tilt: np.ndarray, flat: np.ndarray,
                       heads: Dict[str, np.ndarray]) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "rms", "hnr_proxy", "hf_ratio", "tilt", "flatness", "gate", "strain", "collapse"])
        for i in range(len(times)):
            w.writerow([
                float(times[i]),
                float(rms[i]) if np.isfinite(rms[i]) else "",
                float(hnr[i]) if np.isfinite(hnr[i]) else "",
                float(hf_ratio[i]) if np.isfinite(hf_ratio[i]) else "",
                float(tilt[i]) if np.isfinite(tilt[i]) else "",
                float(flat[i]) if np.isfinite(flat[i]) else "",
                int(bool(heads["gate"][i])),
                float(heads["strain"][i]),
                float(heads["collapse"][i]),
            ])


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline(input_path: str, out_dir: str, cfg: Config) -> Dict[str, Any]:
    ensure_dir(out_dir)

    audio_path = input_path
    if is_video(input_path):
        audio_path = os.path.join(out_dir, "_extracted.wav")
        extract_audio_ffmpeg(input_path, audio_path, cfg.sr)

    y, sr = load_audio_mono(audio_path, cfg.sr)
    y = light_bandpass(y, sr)
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
    flat = spectral_flatness(S)
    hnr = hnr_proxy(y, sr, frame_len, hop_len)

    L = min(len(rms), S.shape[1], len(tilt), len(hf_ratio), len(flat), len(hnr))
    rms, tilt, hf_ratio, flat, hnr = rms[:L], tilt[:L], hf_ratio[:L], flat[:L], hnr[:L]
    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hop_len)

    phrase_mask = make_phrase_frame_mask(times, phrases, sr)

    heads = score_strain_collapse(hf_ratio, tilt, flat, hnr, rms, phrase_mask, cfg)

    # smoothing
    smooth_win = max(3, int(round(cfg.smooth_s / (cfg.hop_ms * 1e-3))))
    strain_s = smooth(heads["strain"], smooth_win)
    collapse_s = smooth(heads["collapse"], smooth_win)

    # regions
    strain_regs = pick_regions(strain_s, times, cfg.strain_thr)
    collapse_regs = pick_regions(collapse_s, times, cfg.collapse_thr)

    regions: List[Dict[str, Any]] = []
    for s, e, p in strain_regs:
        regions.append({"label": "strain", "start_s": float(s), "end_s": float(e), "peak": float(p), "note": coaching_note("strain")})
    for s, e, p in collapse_regs:
        regions.append({"label": "collapse", "start_s": float(s), "end_s": float(e), "peak": float(p), "note": coaching_note("collapse")})

    # edge filter (kills end-of-clip junk)
    regions = drop_edge_regions(regions, audio_dur_s, cfg.edge_ignore_s)

    # top-k only (no duration filter)
    top_strain = top_k(regions, "strain", cfg.top_k_strain)
    top_collapse = top_k(regions, "collapse", cfg.top_k_collapse)

    # scores (light summary)
    scores = {
        "strain_mean": float(np.mean(strain_s)),
        "collapse_mean": float(np.mean(collapse_s)),
    }

    # artifacts
    artifacts = {}
    if cfg.save_timeline:
        timeline_path = os.path.join(out_dir, "timeline.csv")
        write_timeline_csv(timeline_path, times, rms, hnr, hf_ratio, tilt, flat, heads)
        artifacts["timeline_csv"] = os.path.abspath(timeline_path)

    if cfg.save_plot:
        report_path = os.path.join(out_dir, "report.png")
        highlight = sorted(top_strain + top_collapse, key=lambda r: r["start_s"])
        save_report_plot(report_path, y, sr, times, strain_s, collapse_s, highlight)
        artifacts["report_png"] = os.path.abspath(report_path)

    results = {
        "input": os.path.abspath(input_path),
        "audio_used": os.path.abspath(audio_path),
        "sample_rate": cfg.sr,
        "filters": {
            "edge_ignore_s": cfg.edge_ignore_s,
            "phrase_rms_percentile": cfg.phrase_rms_percentile,
            "top_k_strain": cfg.top_k_strain,
            "top_k_collapse": cfg.top_k_collapse,
            "strain_thr": cfg.strain_thr,
            "collapse_thr": cfg.collapse_thr,
        },
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "mistakes": {
            "strain": [
                {"start_s": round(r["start_s"], 3), "end_s": round(r["end_s"], 3), "duration_s": round(r["end_s"]-r["start_s"], 3), "peak": round(r["peak"], 3), "note": r["note"]}
                for r in top_strain
            ],
            "collapse": [
                {"start_s": round(r["start_s"], 3), "end_s": round(r["end_s"], 3), "duration_s": round(r["end_s"]-r["start_s"], 3), "peak": round(r["peak"], 3), "note": r["note"]}
                for r in top_collapse
            ],
        },
        "artifacts": artifacts,
        "disclaimer": "Coaching-oriented acoustic pattern detection, not medical diagnosis."
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Audio (.wav/.mp3/.m4a) OR video (.mp4/.mov/...)")
    p.add_argument("--out_dir", default="bad_singing_out")

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--top_db_split", type=float, default=30.0)
    p.add_argument("--strain_thr", type=float, default=0.62)
    p.add_argument("--collapse_thr", type=float, default=0.62)

    p.add_argument("--top_k_strain", type=int, default=3)
    p.add_argument("--top_k_collapse", type=int, default=3)
    p.add_argument("--edge_ignore_s", type=float, default=1.0)
    p.add_argument("--phrase_rms_percentile", type=float, default=30.0)

    p.add_argument("--no_timeline", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Config(
        sr=args.sr,
        top_db_split=args.top_db_split,
        strain_thr=args.strain_thr,
        collapse_thr=args.collapse_thr,
        top_k_strain=args.top_k_strain,
        top_k_collapse=args.top_k_collapse,
        edge_ignore_s=args.edge_ignore_s,
        phrase_rms_percentile=args.phrase_rms_percentile,
        save_timeline=(not args.no_timeline),
        save_plot=(not args.no_plot),
    )

    res = run_pipeline(args.input, args.out_dir, cfg)

    print("\n=== Top Mistakes (written to results.json) ===")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")

    print("\nSTRAIN:")
    if not res["mistakes"]["strain"]:
        print("  (none)")
    for r in res["mistakes"]["strain"]:
        print(f'  - {r["start_s"]:.2f}s–{r["end_s"]:.2f}s  peak={r["peak"]:.2f}')

    print("\nCOLLAPSE:")
    if not res["mistakes"]["collapse"]:
        print("  (none)")
    for r in res["mistakes"]["collapse"]:
        print(f'  - {r["start_s"]:.2f}s–{r["end_s"]:.2f}s  peak={r["peak"]:.2f}')

    print("\nDone.\n")

if __name__ == "__main__":
    main()
