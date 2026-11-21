"""Floating face bubble and mood FX: pure graph builders and the argv."""
import clip_fx
import floating_face as ff
import mood_fx as mf


def test_head_square_stays_inside_and_even():
    x, y, side = ff.head_square((1800, 900, 100, 120), 1920, 1080)
    assert side % 2 == 0 and x % 2 == 0 and y % 2 == 0
    assert x + side <= 1920 and y + side <= 1080
    assert ff.head_square((0, 0, 5, 5), 1920, 1080) is None
    assert ff.head_square(None, 1920, 1080) is None


def test_median_box():
    assert ff.median_box([(1, 1, 10, 10), (3, 3, 30, 30), (2, 2, 20, 20)]) == (2, 2, 20, 20)
    assert ff.median_box([]) is None


def test_bubble_layout_corners():
    d, x, y = ff.bubble_layout(1080, 1920, "top-right")
    assert x + d < 1080 and y > 0
    d2, x2, _ = ff.bubble_layout(1080, 1920, "top-left")
    assert x2 < x
    _, _, yb = ff.bubble_layout(1080, 1920, "bottom-left")
    assert yb + d < 1920 * 0.6  # stays above the caption band


def test_overlay_graph_is_numbers_only():
    g = ff.overlay_graph((100, 50, 400), 1080, 1920)
    assert "crop=400:400:100:50" in g and g.endswith("[ff]")
    assert "sin(2*PI*t/" in g


def test_mode(monkeypatch):
    monkeypatch.delenv("FLOATING_FACE", raising=False)
    assert ff.mode() == "off"
    monkeypatch.setenv("FLOATING_FACE", "1")
    assert ff.mode() == "on"
    monkeypatch.setenv("FLOATING_FACE", "force")
    assert ff.mode() == "force"


def test_clean_beats_filters_unknown_moods_and_spacing():
    raw = [{"t": 3, "mood": "fun"}, {"t": 4, "mood": "tense"},       # too close
           {"t": 9, "mood": "explode; rm -rf"}, {"t": 0.1, "mood": "sad"},
           {"t": 12, "mood": "HYPE"}, {"t": 99, "mood": "weird"}, {"t": "x", "mood": "fun"}]
    assert mf.clean_beats(raw, 20) == [{"t": 3.0, "mood": "fun"}, {"t": 12.0, "mood": "hype"}]


def test_every_mood_has_sound_and_picture():
    for m in mf.MOODS:
        assert m in mf.SYNTH and m in mf.VISUAL


def test_clip_words_are_relative():
    t = {"segments": [{"words": [{"word": " hi", "start": 10.5}, {"word": "x", "start": 30}]}]}
    assert mf.clip_words(t, 10, 20) == [["hi", 0.5]]


def test_visual_chain_and_audio_graph():
    beats = [{"t": 1.0, "mood": "tense"}, {"t": 6.0, "mood": "fun"}]
    v = mf.visual_chain(beats, 7.0)
    assert "enable='between(t,1.00,3.50)'" in v and "between(t,6.00,7.00)" in v
    a = mf.audio_graph(beats, 2)
    assert "[2:a]" in a and "[3:a]" in a and "adelay=6000|6000" in a
    assert "amix=inputs=3" in a and a.endswith("[mfa]")
    assert mf.audio_graph([], 2) == ""


def test_sfx_dir_override(tmp_path, monkeypatch):
    (tmp_path / "fun.mp3").write_bytes(b"x")
    monkeypatch.setenv("SFX_DIR", str(tmp_path))
    assert mf.sfx_input("fun") == ["-i", str(tmp_path / "fun.mp3")]
    assert mf.sfx_input("sad")[:2] == ["-f", "lavfi"]


def test_build_command_variants(monkeypatch):
    monkeypatch.delenv("SFX_DIR", raising=False)
    assert clip_fx.build_command("c.mp4", "s.mp4", "o.mp4", (1080, 1920), None, [], 10, True) is None
    cmd = clip_fx.build_command("c.mp4", "s.mp4", "o.mp4", (1080, 1920), (0, 0, 400), [], 10, True)
    g = cmd[cmd.index("-filter_complex") + 1]
    assert g.endswith("[vout]") and "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
    beats = [{"t": 2.0, "mood": "weird"}]
    cmd = clip_fx.build_command("c.mp4", "s.mp4", "o.mp4", (1080, 1920), (0, 0, 400), beats, 10, True)
    g = cmd[cmd.index("-filter_complex") + 1]
    assert "[2:a]" in g and "[aout]" in g and "[vout]" in g
    cmd = clip_fx.build_command("c.mp4", "s.mp4", "o.mp4", (1080, 1920), None, beats, 10, False)
    g = cmd[cmd.index("-filter_complex") + 1]
    assert "hue=" in g and "[aout]" not in g


def test_apply_never_raises(monkeypatch):
    monkeypatch.setenv("FLOATING_FACE", "1")
    assert clip_fx.apply("/nope.mp4", "/nope2.mp4", {}, None, 0, 10) is False


def test_detect_uses_the_stage_and_cleans(monkeypatch):
    t = {"segments": [{"words": [{"word": w, "start": i} for i, w in enumerate("a b c d e f".split())]}]}
    stage = lambda client, model, prompt, schema: ({"beats": [{"t": 2, "mood": "fun"},
                                                               {"t": 3, "mood": "nope"}]}, None)
    assert mf.detect(stage, None, "m", t, 0, 10) == [{"t": 2.0, "mood": "fun"}]
