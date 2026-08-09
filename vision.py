"""WALP 2 - camera and body tracking.

The camera and the pose model live on their own thread, so a slow frame or a
hiccup in the tracker can never stall drawing or make the game feel laggy.
The main loop just asks for the most recent result and carries on.

PRIVACY: nothing here writes to disk. No video, no stills, no landmarks.
Frames exist in memory for a few milliseconds and are then dropped.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config
import poses

# Calibration verdicts
CAL_OK = "ok"
CAL_NO_BODY = "no_body"
CAL_TOO_CLOSE = "too_close"
CAL_PARTIAL = "partial"


@dataclass
class VisionFrame:
    landmarks: Optional[list] = None            # primary player
    others: List[list] = field(default_factory=list)   # additional players
    mask: Optional[np.ndarray] = None
    preview: Optional[np.ndarray] = None        # small RGB frame, display only
    frame_index: int = 0
    stamp: float = 0.0


class VisionWorker(threading.Thread):
    """Owns the camera and the pose model exclusively."""

    def __init__(self, num_poses: int = 1, camera_index: int = 0,
                 prefer_dshow: bool = False) -> None:
        super().__init__(daemon=True, name="walp2-vision")
        self.num_poses = num_poses
        self.camera_index = camera_index
        self.prefer_dshow = prefer_dshow

        self._lock = threading.Lock()
        self._frame = VisionFrame()
        self._stop = threading.Event()

        self.ready = threading.Event()
        self.error: Optional[str] = None
        self.status = "starting"
        self.segmentation = False
        self.fps = 0.0

    # -- public ----------------------------------------------------------
    def latest(self) -> VisionFrame:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()

    # -- internals -------------------------------------------------------
    def _build_landmarker(self, mp, model_path: str):
        """Try with the segmentation mask; fall back if the model or the
        installed mediapipe build will not give us one."""
        base = mp.tasks.BaseOptions(model_asset_path=model_path)
        for want_mask in (config.SEGMENTATION_ENABLED, False):
            try:
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=self.num_poses,
                    min_pose_detection_confidence=0.35,
                    min_pose_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                    output_segmentation_masks=want_mask,
                )
                lm = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                self.segmentation = bool(want_mask)
                return lm
            except Exception:
                if not want_mask:
                    raise
                continue
        return None

    def run(self) -> None:
        try:
            import cv2
            import mediapipe as mp
        except Exception as exc:
            self.error = f"Missing library: {exc}"
            self.ready.set()
            return

        import os
        model_path = next((p for p in config.MODEL_CANDIDATES if os.path.exists(p)), None)
        if model_path is None:
            self.error = ("Pose model not found. Expected one of:\n  "
                          + "\n  ".join(config.MODEL_CANDIDATES))
            self.ready.set()
            return

        cap = None
        landmarker = None
        try:
            self.status = "loading model"
            landmarker = self._build_landmarker(mp, model_path)

            self.status = "opening camera"
            # Backend order matters a lot. DirectShow opens in a fraction of a
            # second but can negotiate a 10 fps format on some webcams, while
            # the default (Media Foundation on Windows) is slow to open and
            # then runs at a full 30 fps. Playability beats startup time, so
            # the default goes first and DirectShow is only the fallback.
            backends = [None, getattr(cv2, "CAP_DSHOW", None)]
            if self.prefer_dshow:
                backends.reverse()
            for backend in backends:
                if backend is None:
                    cap = cv2.VideoCapture(self.camera_index)
                else:
                    cap = cv2.VideoCapture(self.camera_index, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
            if cap is None or not cap.isOpened():
                self.error = ("Could not open the camera.\n"
                              "Another program may be using it.")
                self.ready.set()
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.status = "running"
            self.ready.set()

            index = 0
            last_ts = -1
            fps_t0 = time.perf_counter()
            fps_n = 0

            while not self._stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # mediapipe demands strictly increasing timestamps
                ts = int(time.perf_counter() * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                try:
                    result = landmarker.detect_for_video(image, ts)
                except Exception:
                    continue

                found = list(result.pose_landmarks or [])
                mask = None
                if self.segmentation and getattr(result, "segmentation_masks", None):
                    try:
                        raw = result.segmentation_masks[0].numpy_view()
                        d = max(1, config.SEGMENTATION_DOWNSCALE)
                        mask = cv2.resize(raw, (raw.shape[1] // d, raw.shape[0] // d),
                                          interpolation=cv2.INTER_AREA)
                    except Exception:
                        mask = None

                preview = None
                if config.PIP_ENABLED:
                    try:
                        preview = cv2.resize(rgb, (config.PIP_SRC_W, config.PIP_SRC_H),
                                             interpolation=cv2.INTER_AREA)
                    except Exception:
                        preview = None

                index += 1
                with self._lock:
                    self._frame = VisionFrame(
                        landmarks=found[0] if found else None,
                        others=found[1:],
                        mask=mask,
                        preview=preview,
                        frame_index=index,
                        stamp=time.perf_counter(),
                    )

                fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.fps = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now
        except Exception as exc:                       # pragma: no cover
            self.error = f"Camera/tracking error: {exc}"
            self.ready.set()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
            self.status = "stopped"


# --------------------------------------------------------------------------
# Screen-space helpers
# --------------------------------------------------------------------------

class _Point:
    """Stand-in landmark for a position we worked out ourselves."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def _vis(lm) -> float:
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


def to_screen(lm, play) -> Tuple[float, float]:
    """Camera space -> play field. Mirrored, so moving right moves right."""
    return (play.x + play.w - lm.x * play.w, play.y + lm.y * play.h)


def visible_points(landmarks, play) -> dict:
    pts = {}
    if landmarks is None:
        return pts
    for idx in poses.USED_LANDMARKS:
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", None)
        if vis is not None and vis < config.VISIBILITY_THRESHOLD:
            continue
        x, y = to_screen(lm, play)
        pts[idx] = (int(x), int(y))
    return pts


def player_box(landmarks, play, pad: float = 0.10):
    import pygame
    pts = visible_points(landmarks, play)
    if not pts:
        return None
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    px = max(max_x - min_x, 1) * pad
    py = max(max_y - min_y, 1) * pad
    return pygame.Rect(int(min_x - px), int(min_y - py),
                       int(max_x - min_x + px * 2), int(max_y - min_y + py * 2))


def raised_hand_point(landmarks, play) -> Optional[Tuple[float, float]]:
    """The wrist a child is deliberately holding up, used as a mouse pointer.

    Free movement: there is no height gate. See the note in the body for why.
    """
    if landmarks is None:
        return None

    ls, rs = landmarks[poses.L_SHOULDER], landmarks[poses.R_SHOULDER]
    sc_x = (ls.x + rs.x) * 0.5
    sc_y = (ls.y + rs.y) * 0.5
    shoulder_w = max(abs(ls.x - rs.x), 0.03)

    # A hand hanging against your side switches the pointer off; lifting it
    # even slightly switches it on. The bar is set just above the hips, not
    # at the shoulders - a shoulder-height bar is what used to kill the
    # pointer the moment you lowered your arm to reach a button.
    lh, rh = landmarks[poses.L_HIP], landmarks[poses.R_HIP]
    if min(_vis(lh), _vis(rh)) >= 0.3:
        gate_y = (lh.y + rh.y) * 0.5 - shoulder_w * 0.10
    else:
        gate_y = sc_y + shoulder_w * 1.25

    # Of the hands that are up, use whichever is higher / further out - that
    # is the one being aimed. Each hand then covers its own side of the
    # screen instead of making you reach across yourself.
    best = None
    # -inf, not a number: any floor here silently drops the hand once it goes
    # low, which is exactly the "cursor dies when I lower my arm" problem.
    best_reach = float("-inf")
    for wrist_idx, elbow_idx in ((poses.L_WRIST, poses.L_ELBOW),
                                 (poses.R_WRIST, poses.R_ELBOW)):
        w = landmarks[wrist_idx]
        point = w
        if _vis(w) < 0.2 <= _vis(landmarks[elbow_idx]):
            point = landmarks[elbow_idx]        # wrist lost, fall back to elbow
        if point.y >= gate_y:
            continue                            # hand is down by your side
        # Height counts double. Scoring on sideways reach alone let a hand
        # hanging low beat the hand actually being pointed straight ahead,
        # which dragged the pointer to the floor.
        reach = ((sc_y - point.y) / shoulder_w) * 2.0 + abs(point.x - sc_x) / shoulder_w
        if reach > best_reach:
            best, best_reach = point, reach

    if best is None:
        return None

    # Everything is in shoulder widths, so it means the same at any distance.
    dx = (best.x - sc_x) / shoulder_w
    dy = (sc_y - best.y) / shoulder_w      # +1.3 overhead, 0 shoulder, -1.2 at rest

    nx = 0.5 - dx * 0.22                   # mirrored, so it behaves like a mirror
    ny = 0.53 - dy * 0.36                  # arm at rest -> ~0.96, overhead -> ~0.06

    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    return (play.x + nx * play.w, play.y + ny * play.h)


def arms_raised(landmarks) -> Tuple[bool, bool]:
    """(left_arm_up, right_arm_up), as seen on the mirrored screen.

    This is the single most reliable thing the tracker gives us: a plain
    yes/no per side, needing only the wrist and shoulder of that arm. It
    survives distance, low confidence and jitter, none of which a continuous
    hand-position cursor does.
    """
    if landmarks is None:
        return (False, False)
    ls, rs = landmarks[poses.L_SHOULDER], landmarks[poses.R_SHOULDER]
    shoulder_w = max(abs(ls.x - rs.x), 0.03)
    margin = shoulder_w * 0.15

    def up(wrist_idx: int, shoulder) -> bool:
        w = landmarks[wrist_idx]
        return w.y < shoulder.y - margin

    # The play field is mirrored, so the player's own left arm appears on the
    # left of the screen - which is what makes "raise the arm on the side you
    # want to go" feel right.
    return up(poses.L_WRIST, ls), up(poses.R_WRIST, rs)


def check_framing(landmarks):
    """Can we see enough of the player to start? Returns (verdict, advice,
    legs_visible).

    Deliberately only demands head, shoulders and hips. Requiring the whole
    body forced players with a small room into the far corner, where the
    tracking - and the hand cursor with it - falls apart. Leg poses are simply
    never dealt when the legs are out of shot, so playing from the waist up is
    completely fair.
    """
    if landmarks is None:
        return CAL_NO_BODY, "Step in front of the camera", False

    def seen(idx: int, need: float = 0.4) -> bool:
        return _vis(landmarks[idx]) >= need

    head_ok = seen(poses.NOSE, 0.35) and seen(poses.L_SHOULDER) and seen(poses.R_SHOULDER)
    if not head_ok:
        return CAL_NO_BODY, "Step in front of the camera", False

    legs_ok = all(seen(i) for i in (poses.L_KNEE, poses.R_KNEE,
                                    poses.L_ANKLE, poses.R_ANKLE))
    hips_ok = seen(poses.L_HIP) and seen(poses.R_HIP)

    # head cut off at the top is the one framing problem worth blocking on
    nose = landmarks[poses.NOSE]
    if nose.y < 0.02:
        return CAL_TOO_CLOSE, "Step back - your head is cut off", legs_ok

    ls, rs = landmarks[poses.L_SHOULDER], landmarks[poses.R_SHOULDER]
    shoulder_w = abs(ls.x - rs.x)
    if shoulder_w > 0.62:
        return CAL_TOO_CLOSE, "Step back a little", legs_ok
    if shoulder_w < 0.045:
        return CAL_PARTIAL, "Come closer - you are very small in view", legs_ok

    if not hips_ok:
        return CAL_PARTIAL, "Show a bit more of you if you can", legs_ok

    if legs_ok:
        return CAL_OK, "Perfect - the whole body is in view", True
    return CAL_OK, "Good. Legs are out of view, so arm poses only", False
