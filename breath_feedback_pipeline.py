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
        prev = None
        for t in breath_starts:
            if t < bt:
                prev = t
            else:
                break
        # walk backwards until we find something not currently bad
        while prev is not None and prev in bad_set_current:
            # find earlier one
            earlier = None
            for t in breath_starts:
                if t < prev:
                    earlier = t
                else:
                    break
            prev = earlier
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
