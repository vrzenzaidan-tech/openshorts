"""One extra render pass for the opt-in layers: floating face and mood FX.

Runs on the canonical reframe right after it is framed and before the hook
and the captions, so those stay the top layers exactly as before
(``main``'s layer order). A single ffmpeg pass: the face bubble over the
picture, the mood picture effects, and the mood sound effects mixed under the
voice. Whatever goes wrong, the clip stays as it was (the new file replaces
it only after ffmpeg succeeded).
"""
import os
import subprocess

import floating_face
import mood_fx


def wanted() -> bool:
    return floating_face.mode() != "off" or mood_fx.enabled()


def _size(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0:s=x", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return None


def _has_audio(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return bool(out)
    except Exception:
        return False


def build_command(clip_path, source_cut_path, out_path, size, square, beats,
                  duration, has_audio, encode_args=(), tail_args=()):
    """The ffmpeg argv, or None when there is nothing to do."""
    if not square and not beats:
        return None
    w, h = size
    inputs = ["-i", clip_path]
    graph = []
    v = "[0:v]"
    if square:
        inputs += ["-i", source_cut_path]
        graph.append(floating_face.overlay_graph(
            square, w, h, floating_face.position(), video_in=v, face_in="[1:v]", out="[ffo]"))
        v = "[ffo]"
    chain = mood_fx.visual_chain(beats, duration) if beats else ""
    if chain:
        graph.append(f"{v}{chain}[mfv]")
        v = "[mfv]"
    if v == "[0:v]":
        graph.append("[0:v]null[vout]")
    else:
        graph[-1] = graph[-1][: -len(v)] + "[vout]"
    a_map = ["-map", "0:a?"]
    if beats and has_audio:
        first = len(inputs) // 2
        for b in beats:
            inputs += mood_fx.sfx_input(b["mood"])
        graph.append(mood_fx.audio_graph(beats, first, out="[aout]"))
        a_map = ["-map", "[aout]"]
    a_codec = ["-c:a", "aac", "-b:a", "192k"] if a_map == ["-map", "[aout]"] else ["-c:a", "copy"]
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", ";".join(graph), "-map", "[vout]", *a_map,
            *encode_args, *a_codec, *tail_args, "-movflags", "+faststart", out_path]


def apply(clip_path, source_cut_path, clip, transcript, start, end, run_stage=None,
          detect=None):
    """Add the enabled layers to ``clip_path`` in place. Records what it did
    on ``clip`` (``floating_face``, ``mood_beats``). Never raises."""
    if not wanted():
        return False
    try:
        size = _size(clip_path)
        if not size:
            return False
        square = floating_face.plan(source_cut_path, detect)
        beats = []
        if mood_fx.enabled() and transcript and run_stage:
            client, model = mood_fx.model_client()
            if model:
                beats = mood_fx.detect(run_stage, client, model, transcript, start, end)
        from ffmpeg_utils import METADATA_SCRUB, QUALITY, video_encode_args
        tmp = clip_path + ".fx.mp4"
        cmd = build_command(clip_path, source_cut_path, tmp, size, square, beats,
                            float(end) - float(start), _has_audio(clip_path),
                            encode_args=video_encode_args(QUALITY), tail_args=METADATA_SCRUB)
        if not cmd:
            return False
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            print(f"   ⚠️ Clip FX failed, clip kept as it was: {res.stderr[-400:]}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, clip_path)
        clip["floating_face"] = bool(square)
        clip["mood_beats"] = beats
        print(f"   ✨ Clip FX: face={'yes' if square else 'no'}, beats={len(beats)}")
        return True
    except Exception as e:
        print(f"   ⚠️ Clip FX skipped ({type(e).__name__}: {e}).")
        return False
