"""What a YouTube video's own audience points at, as a signal for the moment picker.

Two kinds of evidence a transcript does not carry:

* **Most replayed** (the heatmap above the progress bar). yt-dlp returns it as
  ``info["heatmap"]``: ~100 markers ``{"start_time", "end_time", "value"}``
  with ``value`` normalised to 0..1 by YouTube. The peaks are the stretches
  people scrub back to.
* **Timestamped comments.** "3:45 I lost it", "the 12:01 part 😂": every
  comment that names a time is a viewer pointing at a moment. Mentions are
  counted per moment, weighted by the comment's likes.

Both come out in the same shape as ``twitch_signals.fetch_viewer_clips``
(``{"start", "duration", "views"}`` in source seconds, ``views`` being a
weight), so the picker treats them exactly like Twitch viewer clips: shown to
the scoring and detail passes on the windows that hold them, and the hottest
windows guaranteed a shortlist slot (``twitch_signals.promote_hot_windows``).

Everything here is best effort. Not a YouTube URL, ``YOUTUBE_SIGNALS=0``, no
heatmap (videos with few views have none), comments turned off, a network
error: the job carries on without the signal, exactly as before. The fetch is
anonymous (never ``YOUTUBE_COOKIES``, so it spends nothing from an account's
request budget) and runs inside ``ssrf_guard.guarded()``.

The pure helpers (``heatmap_peaks``, ``parse_timestamps``, ``comment_moments``,
``merge_signals``) are standard-library only and unit-tested.
"""
import os
import re
from typing import Dict, List, Optional

# A marker counts as hot when it is this far up the video's own distribution
# AND clearly above its mean; a flat heatmap (every marker ~equal) has no peak.
HEAT_QUANTILE = 0.85
HEAT_MIN_LIFT = 1.25          # marker value / mean value
HEAT_MAX_PEAKS = 8
HEAT_WEIGHT = 1000            # a full-height (value 1.0) peak's weight

COMMENT_LEAD = 5.0            # seconds before a mentioned time a moment opens
COMMENT_SPAN = 20.0           # and how long it lasts
COMMENT_CLUSTER = 8.0         # mentions this close together are one moment
COMMENT_WEIGHT = 120          # one mention's weight (+ likes, capped)
COMMENT_LIKE_CAP = 200
COMMENT_MAX_MOMENTS = 10

# h:mm:ss or m:ss / mm:ss, not part of a longer number run (dates, scores
# like 3:2:1, ratios inside words). The seconds field is always two digits.
_TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?![\d:])")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def enabled() -> bool:
    return os.environ.get("YOUTUBE_SIGNALS", "1").strip().lower() not in ("0", "false", "no", "off")


def max_comments() -> int:
    try:
        return max(0, int(os.environ.get("YOUTUBE_SIGNAL_COMMENTS", "300")))
    except ValueError:
        return 300


def heatmap_peaks(heatmap, max_peaks: int = HEAT_MAX_PEAKS) -> List[dict]:
    """The "most replayed" peaks as ``[{"start", "duration", "views", "kind"}]``.

    Neighbouring hot markers merge into one stretch; the stretch's weight is
    its highest marker. Strongest first.
    """
    markers = []
    for m in heatmap or []:
        try:
            s, e, v = float(m["start_time"]), float(m["end_time"]), float(m["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s and v >= 0:
            markers.append((s, e, v))
    if len(markers) < 5:
        return []
    markers.sort()
    # YouTube pins the first marker high on most videos (everyone starts at
    # 0:00); it says nothing about a moment.
    body = markers[1:]
    values = sorted(v for _, _, v in body)
    mean = sum(values) / len(values)
    if mean <= 0:
        return []
    cut = values[min(len(values) - 1, int(len(values) * HEAT_QUANTILE))]
    cut = max(cut, mean * HEAT_MIN_LIFT)
    peaks, cur = [], None
    for s, e, v in body:
        if v < cut:
            cur = None
            continue
        if cur and s - cur["end"] <= 0.5:
            cur["end"] = e
            cur["top"] = max(cur["top"], v)
        else:
            cur = {"start": s, "end": e, "top": v}
            peaks.append(cur)
    out = [{"start": round(p["start"], 3), "duration": round(p["end"] - p["start"], 3),
            "views": int(round(p["top"] * HEAT_WEIGHT)), "kind": "replay"}
           for p in peaks]
    out.sort(key=lambda c: -c["views"])
    return out[:max_peaks]


def parse_timestamps(text, duration: Optional[float] = None) -> List[float]:
    """Every time a comment names, in seconds, once each. Times past the end
    of the video (when ``duration`` is known) are not moments of it."""
    seen, out = set(), []
    for h, m, s in _TS.findall(str(text or "")):
        t = int(h or 0) * 3600 + int(m) * 60 + int(s)
        if not h and int(m) > 59 and duration and duration < 3600:
            continue
        if duration and t > duration:
            continue
        if t not in seen:
            seen.add(t)
            out.append(float(t))
    return out


def comment_moments(comments, duration: Optional[float] = None,
                    max_moments: int = COMMENT_MAX_MOMENTS) -> List[dict]:
    """Moments viewers name in comments, ``[{"start", "duration", "views",
    "kind", "mentions"}]``, strongest first.

    A comment that lists many times (a chapter list, "0:00 intro 1:20 ...")
    is an index, not a reaction, and is skipped. Mentions within
    ``COMMENT_CLUSTER`` seconds of each other are one moment.
    """
    hits = []  # (t, weight)
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        times = parse_timestamps(c.get("text"), duration)
        if not times or len(times) > 3:
            continue
        try:
            likes = int(c.get("like_count") or 0)
        except (TypeError, ValueError):
            likes = 0
        w = COMMENT_WEIGHT + min(max(likes, 0), COMMENT_LIKE_CAP)
        hits.extend((t, w) for t in times)
    if not hits:
        return []
    hits.sort()
    clusters = []
    for t, w in hits:
        if clusters and t - clusters[-1]["last"] <= COMMENT_CLUSTER:
            cl = clusters[-1]
            cl["last"] = t
            cl["weight"] += w
            cl["mentions"] += 1
        else:
            clusters.append({"first": t, "last": t, "weight": w, "mentions": 1})
    out = []
    for cl in clusters:
        start = max(0.0, cl["first"] - COMMENT_LEAD)
        end = cl["last"] + COMMENT_SPAN - COMMENT_LEAD
        if duration:
            end = min(end, float(duration))
        if end <= start:
            continue
        out.append({"start": round(start, 3), "duration": round(end - start, 3),
                    "views": int(cl["weight"]), "kind": "comments",
                    "mentions": cl["mentions"]})
    out.sort(key=lambda c: (-c["views"], c["start"]))
    return out[:max_moments]


def merge_signals(*lists) -> List[dict]:
    """One list, strongest first."""
    out = [c for lst in lists for c in (lst or [])]
    out.sort(key=lambda c: -c.get("views", 0))
    return out


def summary(signals: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in signals or []:
        counts[c.get("kind", "?")] = counts.get(c.get("kind", "?"), 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# yt-dlp
# --------------------------------------------------------------------------- #
def _is_youtube(url) -> bool:
    try:
        import source_cache
        return bool(source_cache.video_id(url))
    except Exception:
        return False


def fetch_info(url, n_comments: int, timeout: float = 60.0) -> Optional[dict]:
    """Heatmap, duration and (up to ``n_comments`` top) comments of one video."""
    import yt_dlp
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "noplaylist": True, "socket_timeout": timeout,
        "getcomments": n_comments > 0,
        "extractor_args": {"youtube": {
            "comment_sort": ["top"],
            # max-comments, max-parents, max-replies, max-replies-per-thread
            "max_comments": [str(n_comments), str(n_comments), "0", "0"],
        }},
    }
    import ssrf_guard
    with ssrf_guard.guarded():
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    return info if isinstance(info, dict) else None


def signals_for_job(url) -> List[dict]:
    """Audience signals for this YouTube video in its own seconds (VOD
    seconds; the caller shifts them with ``twitch_signals.shift_clips`` like
    Twitch clips). ``[]`` on anything but a YouTube single-video URL."""
    if not url or not enabled() or not _is_youtube(url):
        return []
    try:
        info = fetch_info(url, max_comments())
    except Exception as e:  # best effort: never fail a job over a hint
        print(f"   ℹ️ YouTube audience signals unavailable ({type(e).__name__}: {e}).")
        return []
    if not info:
        return []
    duration = info.get("duration")
    try:
        duration = float(duration) if duration else None
    except (TypeError, ValueError):
        duration = None
    peaks = heatmap_peaks(info.get("heatmap"))
    talk = comment_moments(info.get("comments"), duration)
    signals = merge_signals(peaks, talk)
    if signals:
        print(f"   🔥 YouTube: {len(peaks)} most-replayed peak(s), "
              f"{len(talk)} moment(s) named in comments.")
    return signals
