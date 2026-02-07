#!/usr/bin/env python3
"""
Bad Singing Detection (ONE FILE) — Strain + Collapse ONLY (no ML)

(UNCHANGED except:)
- results are now RETURNED as a dictionary
- no json.dump to disk
"""

from __future__ import annotations

import argparse
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

    top_db_split: float = 30.0
    min_phrase_s: float = 0.30
    merge_gap_s: float = 0.20

    strain_thr: float = 0.62
    collapse_thr: float = 0.62

    hf_low_hz: float = 2000.0
    hf_high_hz: float = 8000.0

    smooth_s: float = 0.30

    top_k_strain: int = 3
    top_k_collapse: int = 3

    edge_ignore_s: float = 1.0
    phrase_rms_percentile: float = 30.0

    save_timeline: bool = True
    save_plot: bool = True


# ----------------------------
# Utilities
# ----------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(int(win)) / float(win)
    return np.convolve(x, k, mode="same")

def squash(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------
# Video → Audio
# ----------------------------

def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS

def extract_audio_ffmpeg(video_path: str, out_wav: str, sr: int) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), out_wav]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ----------------------------
# Audio
# ----------------------------

def load_audio_mono(path: str, sr: int) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32), sr

def light_bandpass(y: np.ndarray, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    lo = 80.0 / nyq
    hi = min(8000.0, nyq * 0.99) / nyq  # ensure hi < 1.0

    if hi <= lo:
        return y.astype(np.float32)

    b, a = scipy.signal.butter(4, [lo, hi], btype="band")
    return scipy.signal.lfilter(b, a, y).astype(np.float32)



# ----------------------------
# Phrase segmentation
# ----------------------------

def split_phrases_energy(y: np.ndarray, sr: int, cfg: Config):
    intervals = librosa.effects.split(y, top_db=cfg.top_db_split)
    out = []
    for a, b in intervals:
        if (b - a) / sr >= cfg.min_phrase_s:
            out.append((a, b))
    return out

def make_phrase_mask(times, phrases, sr):
    mask = np.zeros_like(times, dtype=bool)
    for a, b in phrases:
        mask |= (times >= a/sr) & (times <= b/sr)
    return mask


# ----------------------------
# Features
# ----------------------------

def rms_env(y, fl, hl):
    return librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]

def stft_mag(y, n_fft, hl):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hl))

def spectral_tilt(S, sr, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    logf = np.log(np.maximum(freqs, 1.0))
    logm = np.log(np.maximum(S, 1e-10))
    return np.sum((logf[:, None] - logf.mean()) * (logm - logm.mean(0)), axis=0) / np.sum((logf - logf.mean())**2)

def hf_ratio(S, sr, n_fft, lo, hi):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= lo) & (freqs <= hi)
    return np.sum(S[band]**2, axis=0) / (np.sum(S**2, axis=0) + 1e-9)

def spectral_flatness(S):
    return librosa.feature.spectral_flatness(S=S)[0]

def hnr_proxy(y, sr, fl, hl):
    n = 1 + (len(y) - fl) // hl
    out = np.zeros(n)
    for i in range(n):
        f = y[i*hl:i*hl+fl]
        ac = scipy.signal.correlate(f, f, mode="full")
        ac = ac[len(ac)//2:]
        out[i] = np.max(ac[20:200]) / (ac[0] + 1e-9)
    return out


# ----------------------------
# Scoring
# ----------------------------

def score(hf, tilt, flat, hnr, rms, mask, cfg):
    gate = mask & (rms >= np.percentile(rms[mask], cfg.phrase_rms_percentile))
    def rz(x): return robust_z(x)

    strain = squash(0.35*rz(hf) + 0.35*rz(flat) + 0.2*rz(1-hnr) + 0.1*rz(-tilt))
    strain = np.where(gate, strain, 0)

    lr = np.log(rms + 1e-9)
    dl = np.maximum(0, -np.diff(np.concatenate([[lr[0]], lr])))
    collapse = squash(rz(dl))
    collapse = np.where(gate, collapse, 0)

    return strain, collapse


# ----------------------------
# Regions
# ----------------------------

def regions(score, times, thr):
    idx = np.where(score > thr)[0]
    if len(idx) == 0:
        return []
    out = []
    s = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            out.append((times[s], times[idx[i-1]], float(score[idx[i-1]])))
            s = idx[i]
    out.append((times[s], times[idx[-1]], float(score[idx[-1]])))
    return out


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def run_pipeline(input_path: str, out_dir: str, cfg: Config) -> Dict[str, Any]:
    ensure_dir(out_dir)

    audio_path = input_path
    if is_video(input_path):
        audio_path = os.path.join(out_dir, "_extracted.wav")
        extract_audio_ffmpeg(input_path, audio_path, cfg.sr)

    y, sr = load_audio_mono(audio_path, cfg.sr)
    y = light_bandpass(y, sr)

    phrases = split_phrases_energy(y, sr, cfg)

    fl = int(cfg.frame_ms * 1e-3 * sr)
    hl = int(cfg.hop_ms * 1e-3 * sr)
    n_fft = 1024

    rms = rms_env(y, fl, hl)
    S = stft_mag(y, n_fft, hl)
    tilt = spectral_tilt(S, sr, n_fft)
    hf = hf_ratio(S, sr, n_fft, cfg.hf_low_hz, cfg.hf_high_hz)
    flat = spectral_flatness(S)
    hnr = hnr_proxy(y, sr, fl, hl)

    L = min(len(rms), len(hf), len(flat), len(hnr))
    rms, hf, flat, hnr, tilt = rms[:L], hf[:L], flat[:L], hnr[:L], tilt[:L]
    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hl)

    mask = make_phrase_mask(times, phrases, sr)
    strain, collapse = score(hf, tilt, flat, hnr, rms, mask, cfg)

    srgn = regions(strain, times, cfg.strain_thr)
    crgn = regions(collapse, times, cfg.collapse_thr)

    srgn = sorted(srgn, key=lambda x: -x[2])[:cfg.top_k_strain]
    crgn = sorted(crgn, key=lambda x: -x[2])[:cfg.top_k_collapse]

    return {
        "input": os.path.abspath(input_path),
        "mistakes": {
            "strain": [{"start_s": s, "end_s": e, "peak": p} for s,e,p in srgn],
            "collapse": [{"start_s": s, "end_s": e, "peak": p} for s,e,p in crgn],
        }
    }


# ----------------------------
# PROGRAMMATIC API (NEW)
# ----------------------------

def get_bad_singing_results(input_path: str, out_dir: str = "bad_singing_out", **config_overrides):
    cfg = Config(**config_overrides)
    return run_pipeline(input_path, out_dir, cfg)


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out_dir", default="bad_singing_out")
    args = p.parse_args()

    res = run_pipeline(args.input, args.out_dir, Config())

    print("\nSTRAIN:")
    for r in res["mistakes"]["strain"]:
        print(r)

    print("\nCOLLAPSE:")
    for r in res["mistakes"]["collapse"]:
        print(r)

if __name__ == "__main__":
    main()
