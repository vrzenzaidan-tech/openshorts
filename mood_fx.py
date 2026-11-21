"""Sound effects and picture effects that follow what happens in a clip.

Gemini (or the local LLM, through the same ``main._run_gemini_stage``) reads
the clip's own words with their times and marks up to ``MAX_BEATS`` beats:
the second something lands and what it feels like, from a closed list
(``MOODS``). Each beat gets:

* a short sound effect, mixed under the voice at that second, and
* a picture effect for ``FX_SECONDS`` (vignette when it gets tense, a colour
  pop when it is fun, a hue wobble when it gets weird, ...).

The sounds are synthesised by ffmpeg itself (``lavfi`` expressions below), so
the repository ships no audio and nothing needs a licence. Drop your own file
at ``$SFX_DIR/<mood>.wav|mp3|ogg|m4a`` to replace any of them.

Off unless ``MOOD_FX=1``. Best effort like every other layer: no key, no words,
an answer that does not parse, an unknown mood: the clip keeps no effects.
The graph builders are pure and unit-tested; the model's answer only ever
reaches the filter graph as a member of ``MOODS`` and as clamped numbers.
"""
import json
import os
from typing import List

from pydantic import BaseModel

MOODS = ("tense", "fun", "confused", "weird", "shocked", "sad", "hype")
MAX_BEATS = 4
MIN_GAP = 4.0             # seconds between two beats
FX_SECONDS = 2.5
SFX_VOLUME = 0.55

# Synthesised effects: (lavfi source, seconds). Plain expressions of t.
SYNTH = {
    # Two low heartbeat thumps per 0.8 s, three beats.
    "tense": ("aevalsrc='0.8*sin(2*PI*52*t)*exp(-9*mod(t,0.8))"
              "+0.6*sin(2*PI*46*t)*exp(-9*mod(t+0.55,0.8))':s=44100:d=2.4", 2.4),
    # Rising "boing".
    "fun": ("aevalsrc='0.5*sin(2*PI*(420+1300*t)*t)*exp(-5*t)':s=44100:d=0.7", 0.7),
    # Falling, wobbling "huh?".
    "confused": ("aevalsrc='0.45*sin(2*PI*(520-260*t+25*sin(2*PI*7*t))*t)"
                 "*exp(-2.2*t)':s=44100:d=1.0", 1.0),
    # Theremin wobble.
    "weird": ("aevalsrc='0.35*sin(2*PI*(380+140*sin(2*PI*5*t))*t)"
              "*(1-exp(-6*t))*exp(-1.2*t)':s=44100:d=1.6", 1.6),
    # A whoosh of noise that swells and cuts.
    "shocked": ("anoisesrc=d=0.8:c=pink:a=0.6:r=44100,"
                "afade=t=in:d=0.6,afade=t=out:st=0.6:d=0.2,lowpass=f=4000", 0.8),
    # Sad trombone: four falling notes with vibrato.
    "sad": ("aevalsrc='0.4*sin(2*PI*(if(lt(t,0.45),311,if(lt(t,0.9),294,"
            "if(lt(t,1.35),277,262+6*sin(2*PI*6*t)))))*t)*exp(-0.6*t)':s=44100:d=2.2", 2.2),
    # Sub drop / impact.
    "hype": ("aevalsrc='0.9*sin(2*PI*(110-70*t)*t)*exp(-2.5*t)':s=44100:d=1.1", 1.1),
}

# Picture effect per mood: ffmpeg filters, each gets enable='between(t,a,b)'.
VISUAL = {
    "tense": ["vignette=angle=PI/3.2", "eq=saturation=0.65:contrast=1.15"],
    "fun": ["eq=saturation=1.5:brightness=0.03"],
    "confused": ["hue=s=0.35", "gblur=sigma=1.6"],
    "weird": ["hue=h=70*sin(2*PI*t*1.5)"],
    "shocked": ["eq=brightness=0.18:contrast=1.2", "vignette=angle=PI/4"],
    "sad": ["eq=saturation=0.3", "colorbalance=bs=0.15:ms=0.08"],
    "hype": ["eq=contrast=1.3:saturation=1.35"],
}

_EXTS = (".wav", ".mp3", ".ogg", ".m4a")


class Beat(BaseModel):
    t: float
    mood: str
    reason: str = ""


class BeatResponse(BaseModel):
    beats: List[Beat]


PROMPT = """You are a short-form video sound designer.
Below are the words of ONE short clip with their times in seconds from the
start of the clip (duration {duration:.1f} s).
Mark at most {max_beats} beats where a sound effect would make the moment land:
the exact second it happens and the feeling, ONE of: {moods}.
- tense: suspense, danger, a risky moment. fun: a joke, a happy surprise.
  confused: someone does not get it, a "wait, what?". weird: bizarre, absurd.
  shocked: a reveal, a big number, an unexpected twist. sad: a letdown, a fail.
  hype: a win, a hype line, a mic drop.
- Fewer is better: only moments that clearly deserve one. Zero is fine.
- Beats at least {gap:.0f} s apart. Never in the first 0.5 s.
WORDS_JSON:
{words_json}
Return only {{"beats": [{{"t": <seconds>, "mood": "<mood>", "reason": "<short>"}}]}}
"""


def enabled() -> bool:
    return os.environ.get("MOOD_FX", "0").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def clip_words(transcript, start, end):
    """``[[word, t], ...]`` with ``t`` in clip seconds."""
    out = []
    for seg in (transcript or {}).get("segments") or []:
        for w in seg.get("words") or []:
            try:
                s = float(w["start"])
            except (KeyError, TypeError, ValueError):
                continue
            if start <= s < end:
                out.append([str(w.get("word", "")).strip(), round(s - start, 2)])
    return out


def clean_beats(beats, duration, max_beats=MAX_BEATS, gap=MIN_GAP) -> List[dict]:
    """Only known moods, inside the clip, ``gap`` apart, time-ordered."""
    ok = []
    for b in beats or []:
        if hasattr(b, "model_dump"):
            b = b.model_dump()
        if not isinstance(b, dict):
            continue
        mood = str(b.get("mood", "")).strip().lower()
        try:
            t = float(b.get("t"))
        except (TypeError, ValueError):
            continue
        if mood not in MOODS or not (0.5 <= t < float(duration) - 0.3):
            continue
        ok.append({"t": round(t, 2), "mood": mood})
    ok.sort(key=lambda b: b["t"])
    out = []
    for b in ok:
        if out and b["t"] - out[-1]["t"] < gap:
            continue
        out.append(b)
        if len(out) >= max_beats:
            break
    return out


def sfx_input(mood) -> List[str]:
    """ffmpeg input arguments for one mood's sound."""
    sfx_dir = os.environ.get("SFX_DIR", "").strip()
    if sfx_dir:
        for ext in _EXTS:
            p = os.path.join(sfx_dir, mood + ext)
            if os.path.isfile(p):
                return ["-i", p]
    src, _ = SYNTH[mood]
    return ["-f", "lavfi", "-i", src]


def visual_chain(beats, duration, fx_seconds=FX_SECONDS) -> str:
    """Comma-joined filters with timeline enables, or "" for none."""
    parts = []
    for b in beats:
        a = max(0.0, float(b["t"]))
        z = min(float(duration), a + fx_seconds)
        for f in VISUAL[b["mood"]]:
            parts.append(f"{f}:enable='between(t,{a:.2f},{z:.2f})'")
    return ",".join(parts)


def audio_graph(beats, first_input, voice_in="[0:a]", out="[mfa]",
                volume=SFX_VOLUME) -> str:
    """Mix beat ``k``'s sound (ffmpeg input ``first_input + k``) at its time."""
    if not beats:
        return ""
    chains, labels = [], [voice_in]
    for k, b in enumerate(beats):
        ms = int(round(float(b["t"]) * 1000))
        lab = f"[mfs{k}]"
        chains.append(f"[{first_input + k}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                      f"volume={volume},adelay={ms}|{ms}{lab}")
        labels.append(lab)
    chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:"
                  f"normalize=0{out}")
    return ";".join(chains)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def detect(run_stage, client, model_name, transcript, start, end) -> List[dict]:
    """The clip's beats, ``[{"t", "mood"}]`` in clip seconds (best effort)."""
    words = clip_words(transcript, start, end)
    if len(words) < 4:
        return []
    duration = float(end) - float(start)
    prompt = PROMPT.format(duration=duration, max_beats=MAX_BEATS, moods=", ".join(MOODS),
                           gap=MIN_GAP, words_json=json.dumps(words, ensure_ascii=False))
    try:
        parsed, _cost = run_stage(client, model_name, prompt, BeatResponse)
    except Exception as e:
        print(f"   🎵 Mood FX: no beats ({type(e).__name__}: {e}).")
        return []
    beats = clean_beats((parsed or {}).get("beats"), duration)
    if beats:
        print("   🎵 Mood FX: " + ", ".join(f"{b['mood']}@{b['t']:.1f}s" for b in beats))
    return beats


def model_client():
    """``(client, model_name)`` the same way the picker chooses them, or
    ``(None, None)`` when no model is configured."""
    import llm_backend
    if llm_backend.active():
        return None, llm_backend.model_name()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None, None
    from google import genai
    return genai.Client(api_key=key), os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"

