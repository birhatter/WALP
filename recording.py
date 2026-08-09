"""WALP - screen recording.

Captures the whole game screen to a video file.

The encoder runs on its own thread behind a small bounded queue. The game
loop only ever does one memcpy of the framebuffer; if the encoder cannot
keep up it drops frames rather than stalling the game. The first version of
WALP encoded full-resolution frames inline on the game loop, which is why it
ran at 33 fps.

NOTE: the recorded screen includes the live camera view, so these files
contain video of whoever is playing. They are written to `recordings/` next
to the game and nothing uploads or shares them.
"""

import os
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np
import pygame

import config


class Recorder:
    """Background screen recorder. Safe to call every frame."""

    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.screen_w, self.screen_h = screen_size

        scale = min(1.0, config.RECORD_MAX_WIDTH / max(self.screen_w, 1))
        self.out_w = int(self.screen_w * scale) // 2 * 2      # even dims for codecs
        self.out_h = int(self.screen_h * scale) // 2 * 2

        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(
            maxsize=config.RECORD_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._writer = None
        self.active = False
        self.path: Optional[str] = None
        self.error: Optional[str] = None
        self.dropped = 0
        self.frames = 0
        self._last_capture = 0.0
        self._interval = 1.0 / max(config.RECORD_FPS, 1)

    # ------------------------------------------------------------------
    def start(self, tag: str = "run") -> bool:
        if self.active:
            return True
        try:
            import cv2
        except ImportError:
            self.error = "opencv-python is not installed"
            return False

        try:
            os.makedirs(config.RECORD_DIR, exist_ok=True)
        except OSError as exc:
            self.error = f"Could not create {config.RECORD_DIR}: {exc}"
            return False

        stamp = time.strftime("%Y%m%d_%H%M%S")
        for codec, ext in (("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
            path = os.path.join(config.RECORD_DIR, f"{tag}_{stamp}{ext}")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec),
                                     config.RECORD_FPS, (self.out_w, self.out_h))
            if writer.isOpened():
                self._writer = writer
                self.path = path
                break
            writer.release()
        else:
            self.error = "No working video codec found"
            return False

        self.error = None
        self.dropped = 0
        self.frames = 0
        self.active = True
        self._last_capture = 0.0
        self._thread = threading.Thread(target=self._encode_loop, daemon=True,
                                        name="walp-recorder")
        self._thread.start()
        return True

    def stop(self) -> Optional[str]:
        """Finish the file and return its path."""
        if not self.active:
            return None
        self.active = False
        try:
            self._queue.put_nowait(None)          # poison pill
        except queue.Full:
            # drain a slot so shutdown is never blocked by a full queue
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except queue.Empty:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        path = self.path
        self.path = None
        return path

    def toggle(self, tag: str = "clip") -> bool:
        if self.active:
            self.stop()
            return False
        return self.start(tag)

    # ------------------------------------------------------------------
    def capture(self, surface: pygame.Surface, now: float) -> None:
        """Grab a frame if one is due. Cheap: one framebuffer copy."""
        if not self.active:
            return
        if now - self._last_capture < self._interval:
            return
        self._last_capture = now
        try:
            raw = pygame.image.tostring(surface, "RGB")
        except pygame.error:
            return
        try:
            self._queue.put_nowait(raw)
        except queue.Full:
            # Encoder is behind. Dropping a frame keeps the game smooth, which
            # matters far more than a perfectly even recording.
            self.dropped += 1

    def _encode_loop(self) -> None:
        import cv2
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                frame = np.frombuffer(item, dtype=np.uint8).reshape(
                    self.screen_h, self.screen_w, 3)
                frame = frame[:, :, ::-1]                     # RGB -> BGR
                if (self.out_w, self.out_h) != (self.screen_w, self.screen_h):
                    frame = cv2.resize(frame, (self.out_w, self.out_h),
                                       interpolation=cv2.INTER_AREA)
                self._writer.write(np.ascontiguousarray(frame))
                self.frames += 1
            except Exception:
                continue
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        self._writer = None


def draw_indicator(surface: pygame.Surface, fonts, recorder: "Recorder",
                   phase: float, topright: Tuple[int, int], scale: float) -> None:
    """Blinking REC dot, so it is never recording without you knowing."""
    if not recorder.active:
        return
    import visuals as V
    x, y = topright
    r = max(5, int(9 * scale))
    if (phase % 1.0) < 0.65:
        pygame.draw.circle(surface, config.INK, (x - r * 2, y + r), r + 2)
        pygame.draw.circle(surface, (255, 60, 80), (x - r * 2, y + r), r)
    V.blit_text(surface, fonts, fonts.small, "REC", (255, 235, 240),
                topright=(x - r * 4, y))
