"""
breath_feedback_pipeline.py

Redistribute "bad breaths" onto previous good breaths.

Inputs:
  - breaths_dict: dict with breaths under breaths_dict["inhalations"] as:
        {"t_start": float, "t_end": float, "duration_s": float, ...}
    NOTE: we only trust t_start.

  - bad_dict: dict with bad regions under bad_dict["mistakes"]["strain"] and ["collapse"] as:
        {"start_s": float, "end_s": float, "peak": float, "note": str}

Core behavior:
  1) Determine "bad breaths" by linking each bad region to a breath (default: nearest BEFORE region start).
  2) For each bad breath, "redistribute" it onto the previous GOOD breath.
  3) Print / return human messages like:
        Replaced bad breath at 12.78s -> previous good breath at 10.02s ...

No transcription. No ML. No ffmpeg. Server-safe.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Parsing helpers
# ----------------------------

def _extract_breath_starts(breaths_dict: Dict[str, Any]) -> List[float]:
    """Return sorted list of breath start times (seconds)."""
    breaths = []
    for b in breaths_dict.get("inhalations", []) or []:
        if "t_start" in b:
            try:
                breaths.append(float(b["t_start"]))
            except Exception:
                pass
    breaths.sort()
    return breaths


def _extract_bad_regions(bad_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten strain+collapse into sorted region list."""
    out: List[Dict[str, Any]] = []
    mistakes = (bad_dict.get("mistakes", {}) or {})
    for label in ("strain", "collapse"):
        for r in (mistakes.get(label, []) or []):
            try:
                out.append({
                    "label": label,
                    "start": float(r["start_s"]),
                    "end": float(r["end_s"]),
                    "peak": float(r.get("peak", 0.0)),
                    "note": str(r.get("note", "")),
                })
            except Exception:
                continue
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


# ----------------------------
# Time search helpers
# ----------------------------

def _nearest_before(times: List[float], t: float) -> Optional[float]:
    """Largest time <= t."""
    lo = None
    for x in times:
        if x <= t:
            lo = x
        else:
            break
    return lo


def _nearest_after(times: List[float], t: float) -> Optional[float]:
    """Smallest time > t."""
    for x in times:
        if x > t:
            return x
    return None


def _pick_anchor_breath(
    breath_starts: List[float],
    region_start: float,
    mode: str = "before",
) -> Tuple[Optional[float], str]:
    """
    Choose which breath "caused" this bad region.

    mode:
      - "before": use nearest breath BEFORE region start; if none, use after
      - "after": use nearest breath AFTER region start; if none, use before
      - "closest": choose closest of before/after
    """
    b = _nearest_before(breath_starts, region_start)
    a = _nearest_after(breath_starts, region_start)

    if mode == "before":
        if b is not None:
            return b, "before"
        return a, "after" if a is not None else (None, "none")

    if mode == "after":
        if a is not None:
            return a, "after"
        return b, "before" if b is not None else (None, "none")

    # closest
    if b is None and a is None:
        return None, "none"
    if b is None:
        return a, "after"
    if a is None:
        return b, "before"
    # both exist
    if abs(region_start - b) <= abs(a - region_start):
        return b, "before"
    return a, "after"


# ----------------------------
# NEW: breathing score helpers
# ----------------------------

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals (assumes seconds)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _time_in_intervals(t: float, intervals: List[Tuple[float, float]]) -> bool:
    for s, e in intervals:
        if s <= t <= e:
            return True
    return False


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _triangular_score(x: float, lo: float, ideal: float, hi: float) -> float:
    """
    0 at <= lo and >= hi.
    1 at ideal.
    Linear ramps between.
    """
    if x <= lo or x >= hi:
        return 0.0
    if x == ideal:
        return 1.0
    if x < ideal:
        return (x - lo) / (ideal - lo)
    return (hi - x) / (hi - ideal)


def calculate_breathing_score(
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    *,
    recovery_window_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Computes the 5-part breathing score:
      Timing (30), Efficiency (25), Rate (20), Support (15), Recovery (10)

    Uses:
      - breaths_dict["inhalations"] with t_start, t_end/duration_s if present
      - breaths_dict["breath_rate_bpm"] if present
      - breaths_dict["breath_rate_timeline"] confidence (fallback support)
      - breaths_dict["quality"] average (fallback support)
      - bad_dict["mistakes"]["strain"/"collapse"] as bad regions
    """
    breath_starts = _extract_breath_starts(breaths_dict)

    inhalations = breaths_dict.get("inhalations", []) or []
    breath_durations = []
    for b in inhalations:
        try:
            if "duration_s" in b and b["duration_s"] is not None:
                d = float(b["duration_s"])
            elif "t_start" in b and "t_end" in b:
                d = float(b["t_end"]) - float(b["t_start"])
            else:
                continue
            if d > 0:
                breath_durations.append(d)
        except Exception:
            continue

    # bad regions as merged intervals
    regions = _extract_bad_regions(bad_dict)
    bad_intervals = _merge_intervals([(r["start"], r["end"]) for r in regions])

    # ----------------------
    # 1) Timing (30%)
    # ----------------------
    if len(breath_starts) == 0:
        timing = 0.0
        timing_detail = {"ok_breaths": 0, "total_breaths": 0}
    else:
        ok = sum(1 for t in breath_starts if not _time_in_intervals(t, bad_intervals))
        timing = ok / len(breath_starts)
        timing_detail = {"ok_breaths": ok, "total_breaths": len(breath_starts)}

    # ----------------------
    # 2) Efficiency (25%)
    # Optimal duration 0.8–2.0s, ideal 1.2s
    # ----------------------
    if len(breath_durations) == 0:
        efficiency = 0.5  # neutral fallback if durations unreliable/missing
        efficiency_detail = {"used": 0, "note": "No reliable duration_s; using neutral fallback."}
    else:
        scores = [_triangular_score(d, 0.8, 1.2, 2.0) for d in breath_durations]
        efficiency = sum(scores) / len(scores)
        efficiency_detail = {"used": len(breath_durations), "avg_duration_s": sum(breath_durations) / len(breath_durations)}

    # ----------------------
    # 3) Rate (20%) - optimal 8–15 BPM
    # ----------------------
    bpm = breaths_dict.get("breath_rate_bpm", None)
    try:
        bpm = float(bpm) if bpm is not None else None
    except Exception:
        bpm = None

    if bpm is None:
        rate = 0.5  # neutral fallback
        rate_detail = {"bpm": None, "note": "No breath_rate_bpm found; using neutral fallback."}
    else:
        # Score peaks around ~12 BPM, drops to 0 at <=8 and >=15
        rate = _triangular_score(bpm, 8.0, 12.0, 15.0)
        rate_detail = {"bpm": bpm}

    # ----------------------
    # 4) Support (15%) - use existing stability metric
    # We'll attempt, in order:
    #   a) average confidence from breath_rate_timeline
    #   b) mean of breaths_dict["quality"] if exists (scaled)
    #   c) fallback neutral
    # ----------------------
    support = None
    support_detail: Dict[str, Any] = {}

    br_tl = breaths_dict.get("breath_rate_timeline", None)
    if isinstance(br_tl, list) and br_tl:
        confs = []
        for w in br_tl:
            try:
                confs.append(float(w.get("confidence", 0.0)))
            except Exception:
                pass
        if confs:
            # confidence is already 0..1-ish
            support = sum(confs) / len(confs)
            support_detail = {"source": "breath_rate_timeline.confidence", "avg_confidence": support}

    if support is None:
        q = breaths_dict.get("quality", None)
        if isinstance(q, list) and q:
            # quality in your dict looks like small values (~0.0003–0.006).
            # We map it to 0..1 by normalizing against a reasonable range.
            q_mean = sum(float(x) for x in q) / len(q)
            # heuristic scaling: 0.0003 -> ~0, 0.006 -> ~1
            support = _clamp01((q_mean - 0.0003) / (0.006 - 0.0003))
            support_detail = {"source": "quality(mean) scaled", "quality_mean": q_mean, "support_scaled": support}

    if support is None:
        support = 0.5
        support_detail = {"source": "fallback", "note": "No stability metric found; using neutral fallback."}

    # ----------------------
    # 5) Recovery (10%)
    # Penalize if bad region starts soon after a breath (within recovery_window_s)
    # ----------------------
    if len(breath_starts) == 0 or len(regions) == 0:
        recovery = 1.0 if len(breath_starts) > 0 else 0.5
        recovery_detail = {"bad_after_breath": 0, "total_breaths": len(breath_starts), "window_s": recovery_window_s}
    else:
        bad_after = 0
        for bt in breath_starts:
            # if ANY bad region begins within (bt, bt+window]
            hit = any((bt < r["start"] <= bt + recovery_window_s) for r in regions)
            if hit:
                bad_after += 1
        recovery = 1.0 - (bad_after / len(breath_starts))
        recovery_detail = {"bad_after_breath": bad_after, "total_breaths": len(breath_starts), "window_s": recovery_window_s}

    # ----------------------
    # Weighted total
    # ----------------------
    weights = {
        "timing": 0.30,
        "efficiency": 0.25,
        "rate": 0.20,
        "support": 0.15,
        "recovery": 0.10,
    }

    total = (
        weights["timing"] * timing +
        weights["efficiency"] * efficiency +
        weights["rate"] * rate +
        weights["support"] * support +
        weights["recovery"] * recovery
    )

    return {
        "subscores": {
            "timing": timing,
            "efficiency": efficiency,
            "rate": rate,
            "support": support,
            "recovery": recovery,
        },
        "details": {
            "timing": timing_detail,
            "efficiency": efficiency_detail,
            "rate": rate_detail,
            "support": support_detail,
            "recovery": recovery_detail,
        },
        "weights": weights,
        "total": total,
    }


# ----------------------------
# Main redistribution logic
# ----------------------------

def redistribute_bad_breaths(
    breaths_dict: Dict[str, Any],
    bad_dict: Dict[str, Any],
    *,
    anchor_mode: str = "before",
    redistribute_to: str = "previous_good",
) -> Dict[str, Any]:
    """
    Core function you import and call.

    anchor_mode:
      - "before" (recommended): bad region blames the breath right before it
      - "after"
      - "closest"

    redistribute_to:
      - "previous_good" (your request): move each bad breath to nearest earlier breath
        that is NOT itself marked bad (after redistribution step-by-step).
    """
    breath_starts = _extract_breath_starts(breaths_dict)
    regions = _extract_bad_regions(bad_dict)
    print(f"--------------------{breath_starts}")

    # Map: anchor_breath_time -> list of region descriptors blamed on it
    blamed: Dict[float, List[Dict[str, Any]]] = {}

    # Track which breath each region anchored to (for transparency)
    region_links: List[Dict[str, Any]] = []

    for r in regions:
        anchor, rel = _pick_anchor_breath(breath_starts, r["start"], mode=anchor_mode)
        region_links.append({
            "region": r,
            "anchor_breath": anchor,
            "anchor_relation": rel,
        })
        if anchor is None:
            continue
        blamed.setdefault(anchor, []).append(r)

    # Initial bad breaths are the keys of `blamed`
    bad_breaths_initial = sorted(blamed.keys())

    # Redistribution:
    # For each bad breath time bt, find previous breath that is currently "good"
    # and move bt's regions there.
    bad_set_current = set(bad_breaths_initial)
    redistributed_from_to: List[Dict[str, Any]] = []
    moved_blame: Dict[float, List[Dict[str, Any]]] = {t: list(v) for t, v in blamed.items()}

    def _find_previous_good(bt: float) -> Optional[float]:
        """
        Find the closest earlier breath such that:
        - t < bt
        - int(t) != int(bt)   (different integer second bucket)
        - t not in bad_set_current
        """
        bt_sec = int(bt)
        prev = None

        # iterate backwards for clarity and correctness
        for t in reversed(breath_starts):
            if t >= bt:
                continue
            if int(t) == bt_sec:
                continue
            if t in bad_set_current:
                continue
            prev = t
            break

        return prev


    if redistribute_to != "previous_good":
        raise ValueError("Only redistribute_to='previous_good' is implemented (as requested).")

    # Process in chronological order for determinism
    for bt in bad_breaths_initial:
        prev_good = _find_previous_good(bt)
        if prev_good is None:
            # cannot redistribute (no earlier good breath)
            redistributed_from_to.append({
                "from": bt,
                "to": None,
                "status": "kept",
                "reason": "no previous good breath available",
            })
            continue

        # Move all blamed regions from bt -> prev_good
        payload = moved_blame.get(bt, [])
        if payload:
            moved_blame.setdefault(prev_good, []).extend(payload)

        # Remove bt blame bucket (optional cleanup)
        moved_blame.pop(bt, None)

        # Update bad set: bt becomes "resolved"; prev_good becomes "bad" now (it carries issues)
        if bt in bad_set_current:
            bad_set_current.remove(bt)
        bad_set_current.add(prev_good)

        redistributed_from_to.append({
            "from": bt,
            "to": prev_good,
            "status": "moved",
            "moved_regions": payload,
        })

    # Build human messages
    messages: List[str] = []
    for item in redistributed_from_to:
        bt = item["from"]
        to = item.get("to")
        if item["status"] == "kept":
            messages.append(
                f"Kept bad breath at {bt:.2f}s (no previous good breath available)."
            )
        else:
            moved_regions = item.get("moved_regions", []) or []
            if moved_regions:
                # summarize first one, but mention count
                first = moved_regions[0]
                extra = ""
                if len(moved_regions) > 1:
                    extra = f" (+{len(moved_regions)-1} more)"
                messages.append(
                    f"Replaced bad breath at {bt:.2f}s → previous good breath at {to:.2f}s "
                    f"({first['label']} @ {first['start']:.2f}–{first['end']:.2f}s){extra}."
                )
            else:
                messages.append(
                    f"Replaced bad breath at {bt:.2f}s → previous good breath at {to:.2f}s."
                )

    # Per-breath summary after redistribution
    per_breath_issues = []
    for t in breath_starts:
        issues = moved_blame.get(t, [])
        if not issues:
            continue
        per_breath_issues.append({
            "breath_time": t,
            "issues": issues,
            "issue_labels": sorted({x["label"] for x in issues}),
            "count": len(issues),
        })

    result = {
        "breaths": breath_starts,
        "bad_regions": regions,
        "anchor_mode": anchor_mode,
        "initial_bad_breaths": bad_breaths_initial,
        "region_links": region_links,  # region -> original anchor breath
        "redistribution": redistributed_from_to,  # from -> to mapping
        "per_breath_issues_after_redistribution": per_breath_issues,
        "messages": messages,  # what you print / show
    }

    # NEW: add breathing score (does not change existing call pattern)
    result["breathing_score"] = calculate_breathing_score(breaths_dict, bad_dict)

    return result


# ----------------------------
# Optional convenience: pretty print
# ----------------------------

def print_redistribution_summary(result: Dict[str, Any]) -> None:
    """Print the human messages."""
    msgs = result.get("messages", []) or []
    if not msgs:
        print("No bad breaths detected (nothing to redistribute).")
        return
    for m in msgs:
        print(m)
