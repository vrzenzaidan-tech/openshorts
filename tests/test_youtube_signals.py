"""YouTube "most replayed" peaks and timestamped comments as picker hints."""
import twitch_signals as ts
import youtube_signals as ys


def _heatmap(values, step=10):
    return [{"start_time": i * step, "end_time": (i + 1) * step, "value": v}
            for i, v in enumerate(values)]


def test_heatmap_peaks_merges_neighbours_and_skips_the_opening_marker():
    values = [1.0] + [0.2] * 20 + [0.9, 0.95] + [0.2] * 20 + [0.7] + [0.2] * 10
    peaks = ys.heatmap_peaks(_heatmap(values))
    assert peaks[0] == {"start": 210.0, "duration": 20.0, "views": 950, "kind": "replay"}
    assert peaks[1]["start"] == 430.0
    assert all(p["start"] > 0 for p in peaks)  # 0:00 spike ignored


def test_flat_or_missing_heatmap_has_no_peaks():
    assert ys.heatmap_peaks(_heatmap([0.5] * 50)) == []
    assert ys.heatmap_peaks(None) == []
    assert ys.heatmap_peaks([{"bad": 1}]) == []


def test_parse_timestamps():
    assert ys.parse_timestamps("lol 3:45 and 1:02:03") == [225.0, 3723.0]
    assert ys.parse_timestamps("at 12:61 nope, 2024-01-01, 10:00:00:00") == []
    assert ys.parse_timestamps("3:45 3:45 again") == [225.0]
    assert ys.parse_timestamps("9:59 past the end", duration=300) == []


def test_comment_moments_cluster_weight_and_skip_chapter_lists():
    comments = [
        {"text": "4:32 ngakak banget 😂", "like_count": 50},
        {"text": "the 4:35 part!!", "like_count": 0},
        {"text": "10:00 meh", "like_count": 0},
        {"text": "0:00 intro 1:00 a 2:00 b 3:00 c", "like_count": 999},  # an index
        {"text": "no time here", "like_count": 10},
    ]
    out = ys.comment_moments(comments, duration=900)
    assert out[0]["start"] == 267.0 and out[0]["mentions"] == 2
    assert out[0]["views"] == 2 * ys.COMMENT_WEIGHT + 50
    assert out[1]["start"] == 595.0
    assert len(out) == 2


def test_signals_flow_through_shift_and_window_heat():
    sig = ys.merge_signals(
        [{"start": 100, "duration": 20, "views": 900, "kind": "replay"}],
        [{"start": 110, "duration": 20, "views": 240, "kind": "comments"}])
    shifted = ts.shift_clips(sig, 0, float("inf"))
    assert shifted[0]["kind"] == "replay"
    ws = [{"id": f"w{i}", "start": i * 90, "end": (i + 1) * 90, "text": ""} for i in range(3)]
    heat = ts.window_heat(ws, shifted)
    assert heat["w1"]["kinds"] == {"replay": 1, "comments": 1}
    assert heat["w1"]["views"] == 1140


def test_non_youtube_or_disabled_returns_nothing(monkeypatch):
    assert ys.signals_for_job("https://www.twitch.tv/videos/123") == []
    monkeypatch.setenv("YOUTUBE_SIGNALS", "0")
    assert ys.signals_for_job("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == []


def test_fetch_failure_is_not_fatal(monkeypatch):
    def boom(url, n):
        raise RuntimeError("offline")
    monkeypatch.setattr(ys, "fetch_info", boom)
    assert ys.signals_for_job("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == []
