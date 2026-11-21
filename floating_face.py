"""A floating face bubble: the speaker's face, cut round, bobbing over the clip.

The case: a wide shot framed GENERAL (blurred backdrop), a screencast, a
gameplay VOD with a small webcam. The person is tiny in the 9:16 frame and the
clip reads as "a video of a screen". A round, bordered face in the top corner,
gently floating, gives the clip a presenter again. Captions stay where they
are (bottom), and are burned after this, on top of everything.

How: sample frames of the clip's own horizontal cut (the ``temp_`` file the
reframe came from, same timeline), find the largest face per frame with
OpenCV's bundled Haar cascade, take the median box and grow it to a square
around the head. That square is cropped from the cut every frame, scaled,
masked to a circle and overlaid with a sine bob. A static box on purpose: a
tracked one wobbles with every detector jitter, and inside a small bubble the
person moving within it reads as natural.

Skipped (returns None, the clip is untouched) when no face is found steadily,
or when the face already fills the source (a talking head: the reframe's
TRACK crop shows it big already), unless ``FLOATING_FACE=force``.

The graph builder is pure and unit-tested; it only ever formats numbers.
"""
import os
from typing import Optional, Tuple

MIN_HIT_RATIO = 0.35      # share of sampled frames that must show a face
BIG_FACE = 0.28           # face height / frame height above which we skip
HEAD_GROW = 2.1           # square side = face height * this (head + shoulders)
SIZE_FRAC = 0.36          # bubble diameter / output width
BORDER_PX = 8
MARGIN_FRAC = 0.05
TOP_FRAC = 0.11           # leave the very top for the platform's own UI
BOB_PX = 14
BOB_PERIOD = 3.2          # seconds per float cycle

POSITIONS = ("top-right", "top-left", "bottom-right", "bottom-left")


def mode() -> str:
    """``off`` (default), ``on`` or ``force``."""
    v = os.environ.get("FLOATING_FACE", "0").strip().lower()
    if v in ("force",):
        return "force"
    return "on" if v in ("1", "true", "yes", "on") else "off"


def position() -> str:
    p = os.environ.get("FLOATING_FACE_POSITION", "top-right").strip().lower()
    return p if p in POSITIONS else "top-right"


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def median_box(boxes):
    """Per-coordinate median of ``[(x, y, w, h), ...]``."""
    if not boxes:
        return None
    cols = list(zip(*boxes))
    mid = len(boxes) // 2
    return tuple(sorted(c)[mid] for c in cols)


def head_square(face, src_w, src_h, grow=HEAD_GROW) -> Optional[Tuple[int, int, int]]:
    """``(x, y, side)`` of a square around a face box, inside the frame.

    The face sits a little above the square's centre so the chin and
    shoulders get room. Even numbers throughout (yuv420 crops)."""
    if not face or src_w <= 0 or src_h <= 0:
        return None
    fx, fy, fw, fh = face
    side = int(min(max(fw, fh) * grow, src_w, src_h))
    side -= side % 2
    if side < 32:
        return None
    cx = fx + fw / 2
    cy = fy + fh / 2 + fh * 0.15
    x = int(min(max(0, cx - side / 2), src_w - side))
    y = int(min(max(0, cy - side / 2), src_h - side))
    return x - x % 2, y - y % 2, side


def bubble_layout(out_w, out_h, where="top-right", size_frac=SIZE_FRAC):
    """``(diameter, x, y)`` of the bubble (border included) in the output."""
    d = int(out_w * size_frac)
    d -= d % 2
    m = int(out_w * MARGIN_FRAC)
    x = out_w - d - m if where.endswith("right") else m
    if where.startswith("top"):
        y = int(out_h * TOP_FRAC)
    else:
        y = int(out_h * 0.58) - d   # above the caption band
    return d, x, y


def overlay_graph(square, out_w, out_h, where="top-right",
                  video_in="[0:v]", face_in="[1:v]", out="[ff]") -> str:
    """filter_complex fragment: the round face (from ``face_in``) floating
    over ``video_in``, labelled ``out``. Numbers only."""
    sx, sy, side = (int(v) for v in square)
    d, x, y = bubble_layout(int(out_w), int(out_h), where)
    inner = d - 2 * BORDER_PX
    r_in = inner / 2.0
    r_out = d / 2.0
    bob = f"{y}+{BOB_PX}*sin(2*PI*t/{BOB_PERIOD})"
    circle_in = f"if(lte(hypot(X-{r_in},Y-{r_in}),{r_in - 1}),255,0)"
    circle_out = f"if(lte(hypot(X-{r_out},Y-{r_out}),{r_out - 1}),255,0)"
    return (
        f"{face_in}crop={side}:{side}:{sx}:{sy},scale={inner}:{inner},format=rgba,"
        f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{circle_in}'[ffface];"
        f"color=c=white:s={d}x{d},format=rgba,"
        f"geq=r='255':g='255':b='255':a='{circle_out}'[ffring];"
        f"{video_in}[ffring]overlay=x={x}:y='{bob}':shortest=1[ffr];"
        f"[ffr][ffface]overlay=x={x + BORDER_PX}:y='{bob}+{BORDER_PX}':shortest=1{out}"
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _haar_detector():
    """OpenCV's bundled Haar cascade, where this OpenCV build still has it."""
    import cv2
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [list(map(int, f)) for f in cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(max(24, frame.shape[0] // 25),) * 2)]
    return detect


def find_face(video_path, detect=None, samples=12):
    """``(face_box, src_w, src_h)`` with the median face, or None.

    ``detect(frame) -> [[x, y, w, h], ...]``: the pipeline passes its own
    MediaPipe detector (``main.detect_face_candidates``); without one the
    Haar cascade is used where available."""
    try:
        import cv2
    except ImportError:
        return None
    detect = detect or _haar_detector()
    if detect is None:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if n <= 0 or w <= 0 or h <= 0:
            return None
        boxes, tried = [], 0
        for k in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            tried += 1
            faces = [f for f in (detect(frame) or []) if len(f) == 4 and f[2] > 0 and f[3] > 0]
            if faces:
                boxes.append(tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3])))
        if not tried or len(boxes) < max(1, tried * MIN_HIT_RATIO):
            return None
        return median_box(boxes), w, h
    finally:
        cap.release()


def plan(source_cut_path, detect=None) -> Optional[Tuple[int, int, int]]:
    """The square to crop from the cut, or None to leave the clip alone."""
    m = mode()
    if m == "off" or not source_cut_path or not os.path.exists(source_cut_path):
        return None
    found = find_face(source_cut_path, detect)
    if not found:
        print("   🙂 Floating face: no steady face in this clip; skipped.")
        return None
    face, w, h = found
    if m != "force" and face[3] / h > BIG_FACE:
        print("   🙂 Floating face: the face already fills the shot; skipped.")
        return None
    return head_square(face, w, h)
