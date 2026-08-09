"""WALP - match the shape, beat the wall.

    python walp2.py                 fullscreen
    python walp2.py --windowed      windowed 1440x810
    python walp2.py --selftest      headless check, no camera needed

Nothing in this game records video, images or body data to disk.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

import config
import poses
import recording
import vision
import visuals as V
import wall as W
from audio import Audio

# ---- modes ---------------------------------------------------------------
LOADING = "loading"
ERROR = "error"
CALIBRATE = "calibrate"
MENU = "menu"
PLAYING = "playing"
PAUSED = "paused"
GAME_OVER = "game_over"
STUDIO = "studio"
COLORS = "colors"

DEFAULT_SAVE = {
    "best": 0, "best_walls": 0, "total_walls": 0,
    "skin": 0, "tutorial_done": False, "muted": False, "camera": 0,
}


def load_save() -> dict:
    data = dict(DEFAULT_SAVE)
    try:
        with open(config.SAVE_PATH, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            for k in DEFAULT_SAVE:
                if k in stored:
                    data[k] = stored[k]
    except (OSError, ValueError):
        pass
    return data


def write_save(data: dict) -> None:
    try:
        with open(config.SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


def unlocked_skins(best: int) -> List[int]:
    return [i for i, (_, _, need) in enumerate(config.PLAYER_SKINS) if best >= need]


@dataclass
class Banner:
    text: str
    color: Tuple[int, int, int]
    duration: float
    age: float = 0.0
    big: bool = True


@dataclass
class FloatScore:
    text: str
    x: float
    y: float
    age: float = 0.0
    duration: float = 1.1
    color: Tuple[int, int, int] = config.SUNNY


class Game:
    def __init__(self, screen: pygame.Surface, worker, audio: Audio,
                 fake_input=None) -> None:
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.scale = self.h / config.DESIGN_H
        self.play = pygame.Rect(0, 0, self.w, self.h)
        self.worker = worker
        self.audio = audio
        self.fake_input = fake_input

        self.fonts = V.Fonts(self.h)
        self.canvas = pygame.Surface((self.w, self.h)).convert()
        self.glow_scratch = pygame.Surface((self.w, self.h), pygame.SRCALPHA).convert_alpha()
        self.clock = pygame.time.Clock()

        self.save = load_save()
        self.audio.muted = bool(self.save.get("muted"))

        self.mode = LOADING
        self.mode_age = 0.0
        self.running = True
        self.phase = 0.0

        self._world_cache: Dict[int, V.World] = {}
        self.world_index = 0
        self.world = self.get_world(0)
        self.p1 = None
        self.p2 = None
        self.newly_unlocked: List[int] = []
        self.stars_played = 0
        self.last_clip = ""
        self.want_record = False
        self.confetti = V.ConfettiSystem(self.scale)
        self.sparkles = V.SparkleSystem()
        self.shake = V.Shake()
        self.transition = V.Transition()
        self.cursor = V.HandCursor(self.w, self.h)
        # no frame around the gauge - asked for, and it keeps the right
        # side of the screen quiet
        self.gauge = V.ScoreGauge(int(self.h * 0.098), panel=False)
        self.wall_renderer = W.WallRenderer()
        self.wall_renderer2 = W.WallRenderer()
        self.loose_shards: List[W.Shard] = []
        self.camera_view = V.CameraView()
        self.recorder = recording.Recorder((self.w, self.h))
        self._bone_colors: Dict = {}

        self.vis = poses.VisibilityTracker()
        self.dealer = poses.PoseDealer(poses.load_custom())

        self.banners: List[Banner] = []
        self.floats: List[FloatScore] = []
        self.buttons: List[V.HoverButton] = []

        # run state
        self.two_player = False
        self.reset_run()

        # calibration
        self.cal_hold = 0.0
        self.cal_verdict = vision.CAL_NO_BODY
        self.cal_advice = ""
        self.cal_legs = False

        # studio
        self.studio_countdown = 0.0
        self.studio_capturing = False
        self.studio_message = ""
        self.studio_message_age = 0.0

        self.debug = False
        self.kb_index = 0
        self.kb_active = False
        self.gesture_hold = 0.0      # how long an arm has been up
        self.gesture_side = 0        # -1 left, +1 right, 0 none
        self.gesture_confirm = 0.0   # how long BOTH arms have been up
        self.gesture_stepped = False
        self.gesture_latched = False

    # ------------------------------------------------------------------
    # run lifecycle
    # ------------------------------------------------------------------
    def reset_run(self) -> None:
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.walls_cleared = 0
        self.lives = config.START_LIVES
        self.bonus_hearts = 0
        self.wall: Optional[W.Wall] = None
        self.spawn_timer = 0.9
        self.live_score: Optional[float] = None
        self.live_score2: Optional[float] = None
        self.smoothed: Optional[float] = None
        self.smoothed2: Optional[float] = None
        self.limb_ema: Dict[Tuple[int, int], float] = {}
        self.joint_scores: Dict[int, float] = {}
        self.heart_break_index: Optional[int] = None
        self.heart_break_age: Optional[float] = None
        self.heart_gain_index: Optional[int] = None
        self.heart_gain_age: Optional[float] = None
        self.label_bounce = 0.0
        self.combo_pop = 0.0
        self.score_pop = 0.0
        self.new_record = False
        self.tutorial_left = 0 if self.save.get("tutorial_done") else 3
        self.confetti.clear()
        self.sparkles.clear()
        self.loose_shards.clear()
        self.banners.clear()
        self.floats.clear()
        self.set_world(0, silent=True)

    def get_world(self, index: int) -> V.World:
        """Themes are expensive to build (gradient + hill bands + sprites), so
        keep a few around. Rebuilding one mid-run causes a visible hitch."""
        index %= len(config.WORLDS)
        world = self._world_cache.get(index)
        if world is None:
            world = V.World(index, self.w, self.h)
            if len(self._world_cache) >= 3:
                self._world_cache.pop(next(iter(self._world_cache)))
            self._world_cache[index] = world
        return world

    def set_world(self, index: int, silent: bool = False) -> None:
        index %= len(config.WORLDS)
        self.world_index = index
        self.world = self.get_world(index)
        self.audio.set_world(index)

    def set_mode(self, mode: str) -> None:
        # Leaving the game for anything except a pause finalises the clip.
        # A video file is only playable once the writer is closed, so a run
        # abandoned through the menu used to leave an unfinished file behind.
        if self.mode == PLAYING and mode not in (PLAYING, PAUSED):
            self.finish_recording()

        self.mode = mode
        self.mode_age = 0.0
        self.buttons = []
        self.kb_index = 0
        if mode == PLAYING:
            self.audio.set_world(self.world_index)
            # Only arm it here. The clip itself starts on the first frame of
            # actual play, once the screen transition has finished, so a
            # recording never opens on the menu or on a wipe.
            self.want_record = True
        else:
            self.audio.play_menu()
        if mode == MENU:
            self.build_menu()
        elif mode == COLORS:
            self.build_colors()
        elif mode == PAUSED:
            self.build_pause()
        elif mode == GAME_OVER:
            self.build_gameover()
        elif mode == STUDIO:
            self.build_studio()

    def go(self, mode: str, color=None) -> None:
        self.transition.start(lambda: self.set_mode(mode),
                              color or self.world.accent)

    # ------------------------------------------------------------------
    # skins
    # ------------------------------------------------------------------
    @property
    def skin_color(self) -> Tuple[int, int, int]:
        idx = self.save.get("skin", 0)
        if not (0 <= idx < len(config.PLAYER_SKINS)):
            idx = 0
        col = config.PLAYER_SKINS[idx][1]
        if col is None:                       # rainbow
            return V.hue_color(self.phase * 0.25)
        return col

    # ------------------------------------------------------------------
    # button screens
    # ------------------------------------------------------------------
    def _row(self, specs, y_frac: float, radius_frac: float = None) -> None:
        r = int(self.h * (radius_frac or config.BUTTON_RADIUS_FRAC))
        n = len(specs)
        for i, (key, label, icon, color) in enumerate(specs):
            cx = int(self.w * 0.5 + (i - (n - 1) / 2) * r * 2.8)
            self.buttons.append(
                V.HoverButton(key, label, (cx, int(self.h * y_frac)), r, color, icon))

    def build_menu(self) -> None:
        self._row([
            ("play", "PLAY", "1", config.MINT),
            ("two", "2 PLAYERS", "2", config.SKY),
            ("poses", "CUSTOM POSES", "+", config.GRAPE),
            ("colors", "COLOUR", "*", config.CORAL),
            ("quit", "QUIT", "x", (150, 146, 168)),
        ], config.BUTTON_ROW_Y)

    def build_pause(self) -> None:
        self._row([
            ("resume", "RESUME", ">", config.MINT),
            ("menu", "MAIN MENU", "^", config.SKY),
            ("quit", "QUIT", "x", (150, 146, 168)),
        ], config.BUTTON_ROW_Y)

    def build_gameover(self) -> None:
        self._row([
            ("play", "PLAY AGAIN", ">", config.MINT),
            ("menu", "MAIN MENU", "^", config.SKY),
        ], 0.78, radius_frac=0.072)

    def build_colors(self) -> None:
        r = int(self.h * 0.075)
        unlocked = unlocked_skins(self.save.get("best", 0))
        n = len(config.PLAYER_SKINS)
        for i, (name, col, need) in enumerate(config.PLAYER_SKINS):
            cx = int(self.w * 0.5 + (i - (n - 1) / 2) * r * 2.6)
            shown = col if col is not None else V.hue_color(i * 0.13)
            b = V.HoverButton(f"skin{i}", name, (cx, int(self.h * 0.52)), r, shown, "")
            b.enabled = i in unlocked
            self.buttons.append(b)
        self.buttons.append(V.HoverButton(
            "back", "BACK", (int(self.w * 0.5), int(self.h * 0.80)),
            int(self.h * 0.07), config.SKY, "^"))

    def build_studio(self) -> None:
        self.buttons.append(V.HoverButton(
            "capture", "CAPTURE", (int(self.w * 0.26), int(self.h * 0.80)),
            int(self.h * 0.075), config.MINT, "!"))
        self.buttons.append(V.HoverButton(
            "clear", "CLEAR ALL", (int(self.w * 0.52), int(self.h * 0.80)),
            int(self.h * 0.075), config.CORAL, "x"))
        self.buttons.append(V.HoverButton(
            "back", "BACK", (int(self.w * 0.78), int(self.h * 0.80)),
            int(self.h * 0.075), config.SKY, "^"))

    def handle_button(self, key: str) -> None:
        self.audio.play("pop")
        if key == "play":
            self.two_player = False
            self.reset_run()
            self.go(PLAYING, config.MINT)
        elif key == "two":
            self.two_player = True
            self.reset_run()
            self.go(PLAYING, config.SKY)
        elif key == "poses":
            self.go(STUDIO, config.GRAPE)
        elif key == "colors":
            self.go(COLORS, config.CORAL)
        elif key == "menu":
            self.go(MENU)
        elif key == "resume":
            self.go(PLAYING, config.MINT)
        elif key == "quit":
            self.running = False
        elif key == "back":
            self.go(MENU)
        elif key == "capture":
            self.studio_countdown = 3.0
            self.studio_capturing = True
        elif key == "clear":
            poses.save_custom([])
            self.dealer = poses.PoseDealer([])
            self.studio_message = "Custom poses cleared"
            self.studio_message_age = 0.0
        elif key.startswith("skin"):
            self.save["skin"] = int(key[4:])
            write_save(self.save)
            self.audio.play("unlock")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, max_frames: Optional[int] = None,
            fixed_dt: Optional[float] = None) -> None:
        frames = 0
        while self.running:
            if fixed_dt is not None:      # selftest: run flat out, no vsync
                self.clock.tick()
                dt = fixed_dt
            else:
                dt = min(self.clock.tick(config.TARGET_FPS) / 1000.0, 0.05)
            self.phase += dt
            self.mode_age += dt
            self.pump_events()
            if not self.running:
                break

            frame = self.get_vision()
            self.update(dt, frame)
            self.render(dt, frame)
            pygame.display.flip()

            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

    def pump_events(self) -> None:
        try:
            events = pygame.event.get()
        except SystemError:
            # The tracking thread's first inference leaves a stale CPython
            # error indicator behind, which surfaces on whichever C call the
            # main thread makes next - usually this one, exactly once, on the
            # frame the camera comes alive. Catching it clears the flag.
            # Dropping a frame of input beats crashing mid-game.
            self._pump_errors = getattr(self, "_pump_errors", 0) + 1
            if self._pump_errors > 50:
                raise
            return

        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_p):
                    if self.mode == PLAYING:
                        self.set_mode(PAUSED)
                    elif self.mode in (PAUSED, GAME_OVER, COLORS, STUDIO):
                        self.set_mode(MENU)
                    elif self.mode == MENU:
                        self.running = False
                elif self.buttons and event.key in (
                        pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    step = 1 if event.key in (pygame.K_RIGHT, pygame.K_DOWN) else -1
                    self.kb_active = True
                    self.move_kb_selection(step)
                elif self.mode == CALIBRATE and event.key in (
                        pygame.K_RETURN, pygame.K_SPACE, pygame.K_KP_ENTER):
                    self.go(MENU)
                elif self.buttons and event.key in (pygame.K_RETURN, pygame.K_SPACE,
                                                    pygame.K_KP_ENTER):
                    self.kb_active = True
                    self.activate_kb_selection()
                elif self.buttons and pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.buttons):
                        self.kb_active = True
                        self.kb_index = idx
                        self.activate_kb_selection()
                elif event.key == pygame.K_r:
                    if self.recorder.toggle("clip"):
                        self.banners.append(Banner("RECORDING", config.CORAL, 1.2))
                    else:
                        self.banners.append(Banner("RECORDING SAVED", config.MINT, 1.4))
                elif event.key == pygame.K_m:
                    self.save["muted"] = self.audio.toggle_mute()
                    write_save(self.save)
                elif event.key == pygame.K_F3:
                    self.debug = not self.debug
                elif event.key == pygame.K_q and (
                        pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.running = False

    def move_kb_selection(self, step: int) -> None:
        usable = [i for i, b in enumerate(self.buttons) if b.enabled]
        if not usable:
            return
        if self.kb_index in usable:
            pos = usable.index(self.kb_index)
            self.kb_index = usable[(pos + step) % len(usable)]
        else:
            self.kb_index = usable[0]
        self.audio.play("hover", 0.5)

    def activate_kb_selection(self) -> None:
        if 0 <= self.kb_index < len(self.buttons):
            b = self.buttons[self.kb_index]
            if b.enabled:
                b.pop = 0.35
                self.handle_button(b.key)

    def get_vision(self) -> vision.VisionFrame:
        if self.fake_input is not None:
            return self.fake_input(self.phase, self.mode)
        return self.worker.latest()

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def players_from(self, frame: vision.VisionFrame):
        """Primary player (biggest body), plus a second one for 2-player."""
        found = []
        if frame.landmarks is not None:
            found.append(frame.landmarks)
        found.extend(frame.others or [])
        if not found:
            return None, None
        if len(found) == 1:
            return found[0], None
        if self.two_player:
            def cx(lms):
                return (lms[poses.L_HIP].x + lms[poses.R_HIP].x) * 0.5
            ordered = sorted(found[:2], key=cx)
            return ordered[0], ordered[1]

        def area(lms):
            xs = [lms[i].x for i in poses.USED_LANDMARKS]
            ys = [lms[i].y for i in poses.USED_LANDMARKS]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
        return max(found, key=area), None

    def update(self, dt: float, frame: vision.VisionFrame) -> None:
        self.transition.update(dt)
        self.world.update(dt)

        p1, p2 = self.players_from(frame)
        self.p1, self.p2 = p1, p2
        self.vis.update(p1)

        hand = vision.raised_hand_point(p1, self.play)
        self.cursor.update(hand, dt)

        for b in self.banners:
            b.age += dt
        self.banners = [b for b in self.banners if b.age < b.duration]
        for f in self.floats:
            f.age += dt
        self.floats = [f for f in self.floats if f.age < f.duration]

        for t in ("heart_break_age", "heart_gain_age"):
            v = getattr(self, t)
            if v is not None:
                setattr(self, t, v + dt)
        self.label_bounce = max(0.0, self.label_bounce - dt)
        self.combo_pop = max(0.0, self.combo_pop - dt)
        self.score_pop = max(0.0, self.score_pop - dt)

        if self.mode == LOADING:
            self.update_loading(dt)
        elif self.mode == CALIBRATE:
            self.update_calibrate(dt, p1)
        elif self.mode == PLAYING:
            self.update_playing(dt, p1, p2)
        elif self.mode == STUDIO:
            self.update_studio(dt, p1)

        if self.mode in (MENU, PAUSED, GAME_OVER, COLORS, STUDIO):
            self.update_buttons(dt)

    def update_gesture_menu(self, dt: float) -> None:
        """Menu control by raising an arm.

        One arm up steps the selection that way; hold it to keep stepping.
        Both arms up selects. Every input here is a plain "is this wrist above
        this shoulder", which the tracker gets right from across a room - the
        continuous hand cursor never did.
        """
        left, right = vision.arms_raised(self.p1)

        # ---- both arms: confirm ----
        if left and right:
            self.gesture_side = 0
            self.gesture_hold = 0.0
            if self.gesture_latched:
                return                       # wait for the arms to come down
            self.gesture_confirm += dt
            if self.gesture_confirm >= config.GESTURE_CONFIRM_SECONDS:
                self.gesture_latched = True
                self.gesture_confirm = 0.0
                self.activate_kb_selection()
            return

        self.gesture_confirm = 0.0
        if not (left or right):
            self.gesture_latched = False

        # ---- one arm: step ----
        side = -1 if left else (1 if right else 0)
        if side == 0:
            self.gesture_side = 0
            self.gesture_hold = 0.0
            self.gesture_stepped = False
            return

        if side != self.gesture_side:
            self.gesture_side = side
            self.gesture_hold = 0.0
            self.gesture_stepped = False

        self.gesture_hold += dt
        due = config.GESTURE_REPEAT if self.gesture_stepped else config.GESTURE_FIRST_STEP
        if self.gesture_hold >= due:
            self.gesture_hold = 0.0
            self.gesture_stepped = True
            self.move_kb_selection(side)

    def update_buttons(self, dt: float) -> None:
        if self.transition.busy:
            return

        if config.GESTURE_MENU:
            # Arm gestures drive the selection outright. Letting the hand
            # cursor also set it meant a single raised arm skipped several
            # buttons at once, because both were writing the same index.
            self.update_gesture_menu(dt)
            for b in self.buttons:
                b.update(self.cursor, dt, focused=False)
            return

        nearest = None
        if config.BUTTON_SNAP and self.cursor.active:
            usable = [b for b in self.buttons if b.enabled]
            if usable:
                cand = min(usable, key=lambda b: math.hypot(
                    self.cursor.pos[0] - b.center[0],
                    self.cursor.pos[1] - b.center[1]))
                reach = cand.radius * 2.6
                if math.hypot(self.cursor.pos[0] - cand.center[0],
                              self.cursor.pos[1] - cand.center[1]) <= reach:
                    nearest = cand
                    self.kb_index = self.buttons.index(nearest)

        for b in self.buttons:
            if b.just_entered:
                self.audio.play("hover", 0.5)
            focused = None if nearest is None else (b is nearest)
            if b.update(self.cursor, dt, focused):
                self.handle_button(b.key)
                return

    def update_loading(self, dt: float) -> None:
        if self.worker is None:
            self.set_mode(CALIBRATE)
            return
        if self.worker.error:
            self.set_mode(ERROR)
        elif self.worker.ready.is_set() and self.mode_age > 0.4:
            self.set_mode(CALIBRATE)

    def update_calibrate(self, dt: float, p1) -> None:
        self.cal_verdict, self.cal_advice, self.cal_legs = vision.check_framing(p1)
        if self.cal_verdict == vision.CAL_OK:
            self.cal_hold += dt
        else:
            self.cal_hold = max(0.0, self.cal_hold - dt * 0.6)

        ready = self.cal_hold >= config.CALIB_REQUIRED_SECONDS
        # Never leave anyone stuck here: as long as we can see a person at
        # all, move on after a while and let the pose dealer handle whatever
        # is actually in view.
        stalled = (self.mode_age > config.CALIB_AUTO_SKIP_SECONDS
                   and self.cal_verdict != vision.CAL_NO_BODY)
        if ready or stalled:
            self.audio.play("go")
            self.go(MENU)

    # -- gameplay ------------------------------------------------------
    def difficulty(self) -> Tuple[float, float, float]:
        n = self.walls_cleared
        if self.tutorial_left > 0:
            return 6.0, 0.48, 1.9
        return (config.travel_seconds(n), config.judgment_start(n),
                config.tolerance_scale(n))

    def begin_recording_for_run(self) -> None:
        if not (config.RECORDING_ENABLED and config.RECORD_AUTO_RUNS):
            return
        if self.recorder.active:
            return
        if not self.recorder.start("run"):
            self.banners.append(Banner("RECORDING FAILED", config.CORAL, 1.8))
            print(f"Recording could not start: {self.recorder.error}")

    def finish_recording(self) -> None:
        """Close the current clip and report where it went."""
        if not self.recorder.active:
            return
        saved = self.recorder.stop()
        if saved:
            self.last_clip = os.path.basename(saved)
            self.banners.append(Banner("CLIP SAVED", config.MINT, 1.4))
            print(f"Saved clip: {saved}")
        elif self.recorder.error:
            print(f"Recording failed: {self.recorder.error}")

    def spawn_wall(self) -> None:
        travel, judge, tol = self.difficulty()
        template = (poses.T_POSE if self.tutorial_left == 3
                    else self.dealer.deal(self.walls_cleared, self.vis))
        spread = config.offcenter_amount(self.walls_cleared)
        x = 0.5 + random.uniform(-spread, spread)
        self.wall = W.Wall(
            template=template,
            color=self.world.accent,
            x_center_frac=x,
            travel_seconds=travel,
            judge_at=judge,
            tolerance=tol,
        )
        self.smoothed = None
        self.smoothed2 = None
        self.limb_ema.clear()
        self.label_bounce = 0.55
        self.audio.play("whoosh", 0.7)
        self.audio.play_pose_cue(template.cue_index)

    def score_player(self, lms, prev: Optional[float]) -> Tuple[Optional[float], Dict[int, float]]:
        if lms is None or self.wall is None:
            return prev, {}
        raw, joints = poses.graded_score(self.wall.template, lms, self.wall.tolerance)
        if prev is None:
            return raw, joints
        return V.lerp(prev, raw, config.LIVE_SCORE_EMA_ALPHA), joints

    def update_playing(self, dt: float, p1, p2) -> None:
        if self.transition.busy:
            return

        if self.want_record:
            self.want_record = False
            self.begin_recording_for_run()

        if self.wall is None:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                self.spawn_wall()
            self.live_score = None
            self.live_score2 = None
            return

        wall = self.wall
        wall.update(dt)

        self.smoothed, self.joint_scores = self.score_player(p1, self.smoothed)
        self.live_score = self.smoothed
        if self.two_player:
            self.smoothed2, _ = self.score_player(p2, self.smoothed2)
            self.live_score2 = self.smoothed2

        if wall.state == W.APPROACHING:
            combined = self.combined_score()
            if wall.judging and combined is not None and combined >= config.PASS_THRESHOLD:
                self.resolve(wall, "hit", combined)
            elif wall.progress >= 1.0:
                s = combined if combined is not None else 0.0
                rescue = config.crack_threshold(self.walls_cleared)
                self.resolve(wall, "crack" if s >= rescue else "miss", s)
            elif wall.judging:
                # ticking urgency while the window is open
                if self.phase - wall.ticked_at > 0.22:
                    wall.ticked_at = self.phase
                    self.audio.play("tick", 0.35)

        if wall.finished:
            self.wall = None
            self.spawn_timer = config.SPAWN_GAP_SECONDS
            self.live_score = None


    def combined_score(self) -> Optional[float]:
        """In 2-player both children have to make the shape, so the pair is
        only as good as whoever is further off. If the camera can only find
        one of them, fall back to solo scoring rather than making the wall
        impossible through no fault of theirs."""
        if not self.two_player:
            return self.live_score
        if self.live_score is None:
            return self.live_score2
        if self.live_score2 is None:
            return self.live_score
        return min(self.live_score, self.live_score2)

    def resolve(self, wall: W.Wall, result: str, score: float) -> None:
        rects = [W.compute_rect(wall, self.play,
                                vision.player_box(lms, self.play),
                                x_center=self.player_x_frac(lms))
                 for lms, _ in self.active_players()]
        if not rects:
            rects = [W.compute_rect(wall, self.play, None)]
        rect = rects[0]
        wall.resolve(result, score)

        if result in ("hit", "crack"):
            self.walls_cleared += 1
            if self.tutorial_left > 0:
                self.tutorial_left -= 1
                if self.tutorial_left == 0:
                    self.save["tutorial_done"] = True
                    write_save(self.save)

            for r in rects:
                self.loose_shards.extend(W.make_shards(
                    wall, r,
                    config.SHARD_COUNT if result == "hit"
                    else config.SHARD_COUNT // 2))
                self.confetti.burst(r.center, wall.color,
                                    52 if result == "hit" else 22)
            self.shake.kick(6 * self.scale, 0.22)
            self.audio.play("shatter", 0.8)
            self.score_pop = 0.45

            if result == "hit":
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                mult = min(self.combo, config.MAX_COMBO_MULTIPLIER)
                gained = config.SCORE_PER_HIT * mult
                self.audio.play("perfect" if score > 0.9 else "hit")
                if self.combo >= 2:
                    self.combo_pop = 0.45
                    self.audio.play_combo(self.combo)
                self.banners.append(Banner(random.choice(config.HIT_WORDS),
                                           config.SUNNY, 1.0))
            else:
                gained = config.SCORE_PER_CRACK
                self.audio.play("crack")
                self.banners.append(Banner(random.choice(config.CRACK_WORDS),
                                           config.SKY, 1.0))

            self.score += gained
            self.floats.append(FloatScore(f"+{gained}", rect.centerx,
                                          rect.centery - 40 * self.scale))
            self.maybe_bonus_heart()
            self.maybe_new_world()
        else:
            self.combo = 0
            self.shake.kick(11 * self.scale, 0.32)
            self.audio.play("miss")
            self.banners.append(Banner(random.choice(config.MISS_WORDS),
                                       config.SKY, 1.0))
            if self.tutorial_left > 0:
                # The first few walls are a lesson, not a test. A child who
                # is still working out what the game wants must not be able
                # to lose hearts, or their very first go ends in failure.
                self.banners.append(Banner("FREE ROUND - NO LIFE LOST", config.MINT, 1.2))
                return
            if self.lives > 0:
                self.heart_break_index = self.lives - 1
                self.heart_break_age = 0.0
                self.lives -= 1
                self.audio.play("heart", 0.8)
            if self.lives <= 0:
                self.end_run()

    def maybe_bonus_heart(self) -> None:
        if self.walls_cleared == 0 or self.walls_cleared % config.BONUS_HEART_EVERY:
            return
        if self.bonus_hearts >= config.MAX_BONUS_HEARTS or self.lives >= config.MAX_LIVES:
            return
        self.bonus_hearts += 1
        self.heart_gain_index = self.lives
        self.heart_gain_age = 0.0
        self.lives += 1
        self.audio.play("unlock")
        self.banners.append(Banner("EXTRA LIFE!", config.CORAL, 1.2))

    def maybe_new_world(self) -> None:
        idx = config.world_index(self.walls_cleared)
        if idx != self.world_index:
            self.set_world(idx)
            self.audio.play("levelup")
            self.banners.append(Banner(self.world.name, self.world.accent, 1.6))

    def end_run(self) -> None:
        before = set(unlocked_skins(self.save.get("best", 0)))

        self.new_record = self.score > self.save.get("best", 0)
        if self.new_record:
            self.save["best"] = self.score
        self.save["best_walls"] = max(self.save.get("best_walls", 0), self.walls_cleared)
        self.save["total_walls"] = self.save.get("total_walls", 0) + self.walls_cleared
        write_save(self.save)

        self.newly_unlocked = sorted(set(unlocked_skins(self.save["best"])) - before)
        self.stars_played = 0

        self.finish_recording()
        self.audio.play("gameover")
        self.go(GAME_OVER, config.GRAPE)

    # -- studio --------------------------------------------------------
    def update_studio(self, dt: float, p1) -> None:
        self.studio_message_age += dt
        if not self.studio_capturing:
            return
        was = self.studio_countdown
        self.studio_countdown -= dt
        if int(was) != int(self.studio_countdown) and self.studio_countdown > 0:
            self.audio.play("countdown")
        if self.studio_countdown > 0:
            return

        self.studio_capturing = False
        existing = poses.load_custom_raw()
        entry = poses.capture_custom(p1, f"MY POSE {len(existing) + 1}")
        if entry is None:
            self.studio_message = "Could not see all of you"
            self.audio.play("miss")
        else:
            existing.append(entry)
            poses.save_custom(existing)
            self.dealer = poses.PoseDealer(poses.load_custom())
            self.studio_message = f"Saved. {len(existing)} custom pose(s)."
            self.audio.play("capture")
            self.confetti.burst((self.w * 0.5, self.h * 0.45), config.GRAPE, 40)
        self.studio_message_age = 0.0

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def pip_rect(self, height_frac: float) -> pygame.Rect:
        h = int(self.h * height_frac)
        w = int(h * config.PIP_SRC_W / config.PIP_SRC_H)
        m = int(self.h * config.PIP_MARGIN_FRAC)
        return pygame.Rect(m, self.h - h - m, w, h)

    def draw_camera(self, frame, height_frac: float) -> None:
        """Live view of the player, bottom left. Display only - this frame
        is drawn and dropped, never written anywhere."""
        if not config.PIP_ENABLED:
            return
        self.camera_view.draw(self.screen, frame.preview, frame.frame_index,
                              self.pip_rect(height_frac), self.world.accent)

    @property
    def player_box(self):
        return vision.player_box(self.p1, self.play) if self.p1 is not None else None

    def render(self, dt: float, frame: vision.VisionFrame) -> None:
        canvas = self.canvas
        self.world.draw(canvas)

        if self.mode == PLAYING:
            self.draw_play_field(canvas, dt, frame)
        elif self.mode in (CALIBRATE, STUDIO):
            self.draw_player(canvas, frame, ghost=None)

        if self.loose_shards:
            self.loose_shards = W.update_shards(self.loose_shards, dt)
            W.draw_shards(canvas, self.loose_shards)
        self.confetti.update_and_draw(canvas, dt)
        self.sparkles.update_and_draw(canvas, dt)

        offset = self.shake.update(dt)
        if offset != (0, 0):
            # only needed to cover the strip the shake exposes
            self.screen.fill(self.world.theme["sky_bottom"])
        self.screen.blit(canvas, offset)

        if self.mode == LOADING:
            self.draw_loading()
        elif self.mode == ERROR:
            self.draw_error()
        elif self.mode == CALIBRATE:
            self.draw_calibrate()
            self.draw_camera(frame, config.PIP_CALIBRATE_HEIGHT_FRAC)
        elif self.mode == MENU:
            self.draw_menu()
            self.draw_camera(frame, config.PIP_HEIGHT_FRAC)
        elif self.mode == PLAYING:
            self.draw_hud()
            self.draw_camera(frame, config.PIP_HEIGHT_FRAC)
        elif self.mode == PAUSED:
            self.draw_paused()
        elif self.mode == GAME_OVER:
            self.draw_gameover()
        elif self.mode == STUDIO:
            self.draw_studio()
            self.draw_camera(frame, config.PIP_HEIGHT_FRAC)
        elif self.mode == COLORS:
            self.draw_colors()

        self.draw_banners()
        if self.mode in (MENU, PAUSED, GAME_OVER, COLORS, STUDIO):
            for i, b in enumerate(self.buttons):
                if i == self.kb_index and b.enabled:
                    pygame.draw.circle(self.screen, config.SUNNY, b.center,
                                       b.radius + int(15 * self.scale),
                                       max(4, int(6 * self.scale)))
                b.draw(self.screen, self.fonts)
            self.draw_aim_prompt()
        if self.mode != PLAYING and not config.GESTURE_MENU:
            self.cursor.draw(self.screen, self.skin_color)

        self.transition.draw(self.screen)
        recording.draw_indicator(self.screen, self.fonts, self.recorder,
                                 self.phase, (self.w - int(24 * self.scale),
                                              int(22 * self.scale)), self.scale)
        self.recorder.capture(self.screen, self.phase)
        if self.debug:
            self.draw_debug()

    # -- play field ----------------------------------------------------
    def draw_play_field(self, canvas, dt: float, frame) -> None:
        y = int(config.JUDGMENT_LINE_Y_FRAC * self.h)
        urgency = 0.0
        if self.wall is not None and self.wall.state == W.APPROACHING:
            urgency = max(0.0, min(1.0, self.wall.progress))
        line_col = V.lerp_color((255, 255, 255), config.SUNNY, urgency)
        dash = int(30 * self.scale)
        for x in range(0, self.w, dash * 2):
            pygame.draw.line(canvas, line_col, (x, y), (x + dash, y),
                             max(3, int(5 * self.scale)))

        self.draw_player(canvas, frame, ghost=None, behind_only=True)

        if self.wall is not None:
            # In two-player each person gets their own wall, sized and centred
            # on them. One shared wall in the middle left both players fighting
            # over the same narrow strip of floor.
            for i, (lms, _) in enumerate(self.active_players()):
                box = vision.player_box(lms, self.play)
                renderer = self.wall_renderer if i == 0 else self.wall_renderer2
                rect = W.compute_rect(self.wall, self.play, box,
                                      x_center=self.player_x_frac(lms))
                score = (self.live_score if i == 0 else self.live_score2) \
                    if self.two_player else self.combined_score()
                renderer.draw(canvas, self.wall, rect, self.phase, score)

        self.draw_player(canvas, frame, ghost=None, outline_only=True)

    def active_players(self):
        """The players a wall should be drawn for."""
        out = []
        if self.p1 is not None:
            out.append((self.p1, self.skin_color))
        if self.two_player and self.p2 is not None:
            out.append((self.p2, config.CORAL))
        return out

    def player_x_frac(self, lms) -> float:
        """Where this player's wall sits across the screen, 0..1.

        Fixed: dead centre for one player, and one half of the screen each for
        two. It deliberately does not track where anyone is standing - the
        walls should arrive in the same place every time so the only thing
        that ever changes is the shape you have to make.
        """
        if not self.two_player or lms is None:
            return config.ONE_PLAYER_WALL_X
        # Only used to work out WHICH half this player is in; the wall then
        # snaps to the centre of that half rather than following them.
        hips = (lms[poses.L_HIP].x + lms[poses.R_HIP].x) * 0.5
        on_screen = 1.0 - hips                     # play field is mirrored
        left, right = config.TWO_PLAYER_WALL_X
        return left if on_screen < 0.5 else right

    def draw_player(self, canvas, frame, ghost=None, behind_only: bool = False,
                    outline_only: bool = False) -> None:
        """Filled body goes behind the wall (so you can see yourself fit
        through the hole); the glowing outline goes on top so a child never
        completely loses track of themselves."""
        skin = self.skin_color

        if not outline_only:
            mask_surf = V.mask_to_surface(frame.mask, (self.w, self.h), skin, 95,
                                          frame.frame_index)
            if mask_surf is not None:
                canvas.blit(mask_surf, (0, 0))

        if behind_only:
            return

        for lms, tint in ((self.p1, skin),
                          (self.p2 if self.two_player else None, config.CORAL)):
            if lms is None:
                continue
            pts = vision.visible_points(lms, self.play)
            if not pts:
                continue
            bone_colors = {}
            if lms is self.p1 and self.wall is not None and self.joint_scores:
                for bone in poses.BONES:
                    vtx = poses.BONE_VERTEX.get(bone)
                    if vtx is None:
                        bone_colors[bone] = tint
                        continue
                    raw = self.joint_scores.get(vtx, 0.0)
                    prev = self.limb_ema.get(bone)
                    raw = raw if prev is None else V.lerp(prev, raw,
                                                          config.LIMB_SCORE_EMA_ALPHA)
                    self.limb_ema[bone] = raw
                    bone_colors[bone] = V.score_to_color(raw)

            head = None
            if poses.NOSE in pts and poses.L_SHOULDER in pts and poses.R_SHOULDER in pts:
                sw = math.hypot(pts[poses.L_SHOULDER][0] - pts[poses.R_SHOULDER][0],
                                pts[poses.L_SHOULDER][1] - pts[poses.R_SHOULDER][1])
                r = max(10, int(sw * 0.46))
                head = ((pts[poses.NOSE][0], pts[poses.NOSE][1] - r // 3), r)

            if lms is self.p1:
                self._bone_colors = bone_colors
            V.draw_player_glow(canvas, pts, poses.BONES, bone_colors, tint, head,
                               max(4, int(9 * self.scale)), self.glow_scratch)

    # -- HUD -----------------------------------------------------------
    def draw_hud(self) -> None:
        s = self.scale
        f = self.fonts

        panel = pygame.Rect(int(22 * s), int(22 * s), int(290 * s), int(70 * s))
        V.draw_panel(self.screen, panel, radius=int(26 * s))
        pop = 1.0 + 0.5 * (self.score_pop / 0.45) if self.score_pop else 1.0
        V.blit_text_scaled(self.screen, f, f.title, str(self.score), config.CREAM,
                           pop, outline=0, midleft=(panel.x + int(22 * s),
                                                    panel.centery))
        V.draw_star(self.screen, (panel.right - int(30 * s), panel.centery),
                    int(19 * s), config.SUNNY)
        V.draw_hearts(self.screen, self.lives, config.MAX_LIVES,
                      (int(26 * s), panel.bottom + int(10 * s)), int(19 * s),
                      self.heart_break_index, self.heart_break_age,
                      self.heart_gain_index, self.heart_gain_age)

        if self.wall is not None:
            tmpl = self.wall.template
            bounce = 1.0 + 0.35 * math.sin(math.pi * (1 - self.label_bounce / 0.55)) \
                if self.label_bounce else 1.0
            pw, ph = int(470 * s), int(108 * s)
            p = pygame.Rect(self.w // 2 - pw // 2, int(16 * s), pw, ph)
            V.draw_panel(self.screen, p, radius=int(30 * s))
            ib = pygame.Rect(0, 0, int(ph * 0.78 * tmpl.box_aspect), int(ph * 0.78))
            ib.center = (p.x + int(62 * s), p.centery)
            W.draw_pose_icon(self.screen, tmpl, ib, self.world.accent)
            V.blit_text_scaled(self.screen, f, f.label, tmpl.label, config.CREAM,
                               bounce, outline=0,
                               center=(p.x + int(pw * 0.61), p.centery))

        gx, gy = int(self.w * 0.895), int(self.h * 0.34)
        if self.two_player:
            self.gauge.draw(self.screen, f, self.live_score, self.phase,
                            (gx, int(self.h * 0.26)))
            V.blit_text(self.screen, f, f.small, "P1", (255, 255, 255),
                        midtop=(gx, int(self.h * 0.26) + int(self.h * 0.14)))
            self.gauge.draw(self.screen, f, self.live_score2, self.phase,
                            (gx, int(self.h * 0.58)))
            V.blit_text(self.screen, f, f.small, "P2", (255, 255, 255),
                        midtop=(gx, int(self.h * 0.58) + int(self.h * 0.14)))
        else:
            self.gauge.draw(self.screen, f, self.live_score, self.phase, (gx, gy))

        if self.combo >= 2:
            scale = 1.0 + 0.7 * (self.combo_pop / 0.45) if self.combo_pop else 1.0
            V.blit_text_scaled(self.screen, f, f.title,
                               f"x{min(self.combo, config.MAX_COMBO_MULTIPLIER)}",
                               config.SUNNY, scale,
                               center=(gx, gy + int(self.h * 0.20)))


        self.draw_pose_guide_panel()

        if self.tutorial_left > 0:
            V.blit_text(self.screen, f, f.mid, "Match the shape in the wall",
                        (255, 255, 255), midbottom=(self.w // 2, self.h - int(24 * s)))

        for fl in self.floats:
            t = fl.age / fl.duration
            V.blit_text_scaled(self.screen, f, f.title, fl.text, fl.color,
                               1.0, alpha=int(255 * (1 - t)),
                               center=(int(fl.x), int(fl.y - 110 * s * t)))

    def draw_pose_guide_panel(self) -> None:
        """Bottom right: the shape you are meant to be making, with your own
        limbs colour-coded. Drawn bare, with no panel behind it, so it stays
        out of the way."""
        if self.wall is None:
            return
        s = self.scale
        tmpl = self.wall.template
        m = int(self.h * config.PIP_MARGIN_FRAC)
        fig_h = int(self.h * 0.155)
        fig = pygame.Rect(0, 0, int(fig_h * tmpl.box_aspect), fig_h)
        fig.bottomright = (self.w - m, self.h - m - int(26 * s))
        W.draw_pose_guide(self.screen, tmpl, fig, self._bone_colors,
                          base=(236, 240, 255))
        V.blit_text(self.screen, self.fonts, self.fonts.tiny, "MATCH THIS",
                    (206, 204, 232), outline=4,
                    midtop=(fig.centerx, fig.bottom + int(4 * s)))

    def draw_aim_prompt(self) -> None:
        """Tell people how to drive the menu, and say so loudly when we cannot
        see a hand at all - nobody should be left waving at a dead screen."""
        if self.transition.busy:
            return
        s = self.scale
        y = int(self.h * 0.90)

        if config.GESTURE_MENU:
            left, right = vision.arms_raised(self.p1)
            lit, dim = config.SUNNY, (176, 174, 204)
            V.blit_text(self.screen, self.fonts, self.fonts.mid, "< LEFT ARM",
                        lit if left else dim, outline=5,
                        midright=(int(self.w * 0.44), y))
            V.blit_text(self.screen, self.fonts, self.fonts.mid, "RIGHT ARM >",
                        lit if right else dim, outline=5,
                        midleft=(int(self.w * 0.56), y))
            V.blit_text(self.screen, self.fonts, self.fonts.small,
                        "BOTH ARMS = SELECT",
                        config.MINT if (left and right) else dim, outline=4,
                        center=(self.w // 2, y + int(48 * s)))
            return

        if self.cursor.active:
            V.blit_text(self.screen, self.fonts, self.fonts.small,
                        "hold over a button to choose", (196, 194, 220),
                        center=(self.w // 2, y))
        else:
            pulse = 0.5 + 0.5 * math.sin(self.phase * 3.4)
            V.blit_text_scaled(self.screen, self.fonts, self.fonts.mid,
                               "LIFT A HAND TO AIM", config.SUNNY,
                               1.0 + 0.04 * pulse, outline=5,
                               center=(self.w // 2, y))
            V.blit_text(self.screen, self.fonts, self.fonts.small,
                        "arrow keys and ENTER also work", (196, 194, 220),
                        center=(self.w // 2, y + int(46 * s)))

    def draw_banners(self) -> None:
        for i, b in enumerate(self.banners):
            t = b.age / b.duration
            scale = V.ease_out_back(min(1.0, t * 3.2)) if t < 0.35 else 1.0
            alpha = 255 if t < 0.7 else int(255 * (1 - (t - 0.7) / 0.3))
            font = self.fonts.big if b.big else self.fonts.mid
            V.blit_text_scaled(self.screen, self.fonts, font, b.text, b.color,
                               scale, outline=6, alpha=max(0, alpha),
                               center=(self.w // 2,
                                       int(self.h * 0.36) + i * int(90 * self.scale)))

    # -- screens -------------------------------------------------------
    def _title(self, text: str, y_frac: float = 0.12, color=config.SUNNY) -> None:
        V.blit_text(self.screen, self.fonts, self.fonts.huge, text, color,
                    outline=7, center=(self.w // 2, int(self.h * y_frac)))

    TIPS = ("Stand back until your whole body is in frame",
            "Match the shape before the wall reaches you",
            "Limbs turn green when they are in position",
            "A near miss scrapes you through at half points",
            "Raise a hand to aim at the menu buttons",
            "Walls keep getting faster. They never stop.")

    def draw_loading(self) -> None:
        self._title(config.GAME_TITLE)
        msg = "Starting camera..."
        if self.worker is not None and self.worker.status:
            msg = f"{self.worker.status.capitalize()}..."
        V.blit_text(self.screen, self.fonts, self.fonts.mid, msg, (255, 255, 255),
                    center=(self.w // 2, int(self.h * 0.30)))

        # A pose parade keeps the wait interesting and teaches the shapes
        # before the child has even started.
        i = int(self.phase * 0.8)
        tmpl = poses.PRESET_TEMPLATES[i % len(poses.PRESET_TEMPLATES)]
        bh = int(self.h * 0.30)
        box = pygame.Rect(0, 0, int(bh * tmpl.box_aspect), bh)
        box.center = (self.w // 2, int(self.h * 0.52))
        W.draw_pose_icon(self.screen, tmpl, box, config.SUNNY)
        V.blit_text(self.screen, self.fonts, self.fonts.small, tmpl.label,
                    (255, 255, 255), midtop=(self.w // 2, box.bottom + int(14 * self.scale)))

        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    self.TIPS[int(self.phase / 3.0) % len(self.TIPS)],
                    config.MINT, center=(self.w // 2, int(self.h * 0.78)))

        r = int(self.h * 0.045)
        V.draw_progress_ring(self.screen, (self.w // 2, int(self.h * 0.90)), r,
                             (self.phase * 0.55) % 1.0, config.MINT,
                             max(6, int(r * 0.3)))

    def draw_error(self) -> None:
        self._title("OOPS!", color=config.CORAL)
        msg = (self.worker.error if self.worker and self.worker.error
               else "Something went wrong.")
        for i, line in enumerate(str(msg).split("\n")[:6]):
            V.blit_text(self.screen, self.fonts, self.fonts.small, line,
                        (255, 255, 255),
                        center=(self.w // 2, int(self.h * 0.42) + i * int(42 * self.scale)))
        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    "Press ESC to close", config.SUNNY,
                    center=(self.w // 2, int(self.h * 0.8)))

    def draw_calibrate(self) -> None:
        self._title("FRAME YOURSELF")
        ok = self.cal_verdict == vision.CAL_OK
        col = config.MINT if ok else config.SUNNY
        V.blit_text(self.screen, self.fonts, self.fonts.big, self.cal_advice, col,
                    center=(self.w // 2, int(self.h * 0.30)))

        # a body-shaped frame to stand inside
        bw, bh = int(self.h * 0.30), int(self.h * 0.52)
        box = pygame.Rect(self.w // 2 - bw // 2, int(self.h * 0.36), bw, bh)
        pygame.draw.rect(self.screen, col, box, width=max(4, int(6 * self.scale)),
                         border_radius=int(30 * self.scale))
        W.draw_ghost_target(self.screen, poses.T_POSE, box, col, 120)

        r = int(self.h * 0.075)
        V.draw_progress_ring(self.screen, (self.w // 2, int(self.h * 0.93) - r), r,
                             self.cal_hold / config.CALIB_REQUIRED_SECONDS,
                             config.MINT, max(7, int(r * 0.28)))
        V.blit_text(self.screen, self.fonts, self.fonts.small,
                    "Hold still" if ok else "Head and shoulders is enough",
                    (255, 255, 255),
                    center=(self.w // 2, int(self.h * 0.93) - r))
        if self.mode_age > 5.0:
            V.blit_text(self.screen, self.fonts, self.fonts.small,
                        "Press ENTER to skip", (196, 194, 220),
                        midtop=(self.w // 2, int(self.h * 0.22)))

    def draw_menu(self) -> None:
        self._title(config.GAME_TITLE)
        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    "Match the shape. Beat the wall.", (255, 255, 255),
                    center=(self.w // 2, int(self.h * 0.24)))
        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    "Raise a hand to aim, hold over a button to select",
                    config.SUNNY, center=(self.w // 2, int(self.h * 0.34)))
        best = self.save.get("best", 0)
        V.blit_text(self.screen, self.fonts, self.fonts.small,
                    f"BEST {best}    WALLS SMASHED {self.save.get('total_walls', 0)}",
                    (255, 255, 255), center=(self.w // 2, int(self.h * 0.86)))
        if self.audio.muted:
            V.blit_text(self.screen, self.fonts, self.fonts.tiny, "MUTED (M)",
                        config.CORAL, topright=(self.w - 20, 20))

    def draw_paused(self) -> None:
        veil = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        veil.fill((20, 14, 40, 170))
        self.screen.blit(veil, (0, 0))
        self._title("PAUSED", y_frac=0.22, color=config.CREAM)
        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    f"Score {self.score}", config.SUNNY,
                    center=(self.w // 2, int(self.h * 0.36)))

    def draw_gameover(self) -> None:
        s = self.scale
        f = self.fonts
        self._title(random.Random(self.walls_cleared).choice(config.GAMEOVER_WORDS),
                    y_frac=0.13, color=config.CREAM)

        y = int(self.h * 0.28)
        V.blit_text(self.screen, f, f.big, f"{self.score}", config.SUNNY,
                    center=(self.w // 2, y))
        V.blit_text(self.screen, f, f.small, "SCORE", (255, 255, 255),
                    center=(self.w // 2, y + int(56 * s)))

        stars = sum(1 for t in config.STAR_THRESHOLDS if self.walls_cleared >= t)
        sy = int(self.h * 0.46)
        for i in range(3):
            sx = self.w // 2 + (i - 1) * int(96 * s)
            lit = i < stars
            pop = 1.0
            if lit:
                local = self.mode_age - i * 0.35
                if local < 0:
                    continue
                pop = 1.0 + 0.7 * max(0.0, 1 - local / 0.3)
                if self.stars_played <= i:
                    self.stars_played = i + 1
                    self.audio.play("star")
            V.draw_star(self.screen, (sx, sy), int(40 * s * pop),
                        config.SUNNY if lit else (120, 116, 140))

        info = [f"WALLS SMASHED   {self.walls_cleared}",
                f"BEST COMBO   x{min(self.best_combo, config.MAX_COMBO_MULTIPLIER)}"]
        if self.new_record:
            info.append("NEW RECORD!")
        else:
            info.append(f"BEST   {self.save.get('best', 0)}")
        for i, line in enumerate(info):
            col = config.CORAL if line == "NEW RECORD!" else (255, 255, 255)
            V.blit_text(self.screen, f, f.mid, line, col,
                        center=(self.w // 2, int(self.h * 0.58) + i * int(52 * s)))

        if self.last_clip:
            V.blit_text(self.screen, f, f.tiny, f"clip saved: {self.last_clip}",
                        (170, 168, 200),
                        midbottom=(self.w // 2, self.h - int(14 * s)))

        if self.newly_unlocked:
            name = config.PLAYER_SKINS[self.newly_unlocked[0]][0]
            V.blit_text_scaled(self.screen, f, f.mid,
                               f"UNLOCKED: {name.upper()} GLOW", config.MINT,
                               1.0 + 0.06 * math.sin(self.phase * 6),
                               center=(self.w // 2, int(self.h * 0.70)))

    def draw_colors(self) -> None:
        self._title("GLOW COLOUR")
        best = self.save.get("best", 0)
        for i, b in enumerate(self.buttons):
            if not b.key.startswith("skin"):
                continue
            idx = int(b.key[4:])
            need = config.PLAYER_SKINS[idx][2]
            if not b.enabled:
                V.blit_text(self.screen, self.fonts, self.fonts.tiny,
                            f"{need} pts", (255, 255, 255),
                            midtop=(b.center[0], b.center[1] + b.radius + int(48 * self.scale)))
            elif idx == self.save.get("skin", 0):
                pygame.draw.circle(self.screen, config.SUNNY, b.center,
                                   b.radius + int(16 * self.scale),
                                   max(4, int(7 * self.scale)))
        V.blit_text(self.screen, self.fonts, self.fonts.small,
                    f"BEST {best} - keep playing to unlock the rest",
                    (255, 255, 255), center=(self.w // 2, int(self.h * 0.66)))

    def draw_studio(self) -> None:
        self._title("CUSTOM POSES")
        V.blit_text(self.screen, self.fonts, self.fonts.mid,
                    "Hold any shape, then capture it", (255, 255, 255),
                    center=(self.w // 2, int(self.h * 0.22)))
        if self.studio_capturing:
            n = max(1, int(math.ceil(self.studio_countdown)))
            V.blit_text_scaled(self.screen, self.fonts, self.fonts.huge, str(n),
                               config.SUNNY,
                               1.0 + 0.5 * (self.studio_countdown % 1.0),
                               outline=7, center=(self.w // 2, int(self.h * 0.45)))
        custom = self.dealer.custom
        V.blit_text(self.screen, self.fonts, self.fonts.small,
                    f"{len(custom)} custom pose(s) saved",
                    (255, 255, 255), center=(self.w // 2, int(self.h * 0.63)))
        box_w = int(self.h * 0.11)
        for i, t in enumerate(custom[:8]):
            bx = int(self.w * 0.5 + (i - (min(len(custom), 8) - 1) / 2) * box_w * 1.35)
            r = pygame.Rect(0, 0, int(box_w * t.box_aspect), box_w)
            r.center = (bx, int(self.h * 0.71))
            W.draw_pose_icon(self.screen, t, r, config.GRAPE)
        if self.studio_message and self.studio_message_age < 3.0:
            V.blit_text(self.screen, self.fonts, self.fonts.mid, self.studio_message,
                        config.MINT, center=(self.w // 2, int(self.h * 0.55)))

    def draw_debug(self) -> None:
        fps = self.clock.get_fps()
        cam = self.worker.fps if self.worker else 0.0
        hand = "-"
        if self.p1 is not None:
            pt = vision.raised_hand_point(self.p1, self.play)
            hand = "none" if pt is None else f"{pt[0] / self.w:.2f},{pt[1] / self.h:.2f}"
        lines = [f"fps {fps:4.1f}  cam {cam:4.1f}",
                 f"mode {self.mode} walls {self.walls_cleared}",
                 f"body {self.p1 is not None}  2nd {self.p2 is not None}",
                 f"legs {self.vis.sees_legs()} arms {self.vis.sees_arms()}",
                 f"hand {hand}  cursor {self.cursor.active}",
                 f"rec {self.recorder.active} frames {self.recorder.frames} "
                 f"dropped {self.recorder.dropped}",
                 f"seg {getattr(self.worker, 'segmentation', False)}"]
        if self.wall:
            lines.append(f"prog {self.wall.progress:.2f} judge {self.wall.judge_at:.2f} "
                         f"tol {self.wall.tolerance:.2f}")
        for i, line in enumerate(lines):
            V.blit_text(self.screen, self.fonts, self.fonts.tiny, line,
                        (255, 255, 255), outline=2, topleft=(12, self.h - 210 + i * 26))
def list_cameras(limit: int = 6) -> int:
    """Probe camera indexes so an external webcam can be found and selected."""
    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed")
        return 1

    print("Probing cameras...\n")
    found = []
    for idx in range(limit):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        size = ""
        if ok:
            got, frame = cap.read()
            if got and frame is not None:
                size = f"  {frame.shape[1]}x{frame.shape[0]}"
                found.append(idx)
            else:
                ok = False
        cap.release()
        print(f"  --camera {idx}   {'WORKS' if ok else 'not available'}{size}")

    print()
    if not found:
        print("No cameras found. Check it is plugged in and not in use.")
        return 1
    print("Usable indexes:", ", ".join(str(i) for i in found))
    print(f"An external webcam is usually the highest number ({found[-1]}).")
    print(f"Start the game with:  python walp2.py --camera {found[-1]}")
    print("That choice is remembered, so you only pass it once.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WALP 2")
    ap.add_argument("--windowed", action="store_true", help="run in a window")
    ap.add_argument("--selftest", action="store_true", help="headless check")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index (0 = built-in, 1+ = external). "
                         "Remembered for next time.")
    ap.add_argument("--list-cameras", action="store_true",
                    help="show which camera indexes work, then exit")
    ap.add_argument("--dshow", action="store_true",
                    help="prefer the DirectShow camera backend (faster to "
                         "start, slower on some webcams)")
    args = ap.parse_args()

    if args.selftest:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "dev"))
        try:
            import selftest as dev_selftest
        except ImportError:
            print("The self test lives in dev/selftest.py, which is a "
                  "developer tool and is not shipped with the game.")
            return 1
        return dev_selftest.selftest()

    if args.list_cameras:
        return list_cameras()

    saved = load_save()
    camera_index = args.camera if args.camera is not None else int(
        saved.get("camera", 0))
    if args.camera is not None and args.camera != saved.get("camera"):
        saved["camera"] = args.camera
        write_save(saved)

    pygame.init()
    if args.windowed:
        screen = pygame.display.set_mode((1440, 810))
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption(config.GAME_TITLE)
    pygame.mouse.set_visible(False)

    # Start the camera thread and put something on screen before synthesising
    # the audio, which takes a second or two of pure numpy work.
    if config.RECORDING_ENABLED and config.RECORD_CREATE_DIR_ON_START:
        try:
            os.makedirs(config.RECORD_DIR, exist_ok=True)
            print(f"Clips will be saved to: {config.RECORD_DIR}")
        except OSError as exc:
            print(f"Could not create {config.RECORD_DIR}: {exc}")

    worker = vision.VisionWorker(num_poses=2, camera_index=camera_index,
                                 prefer_dshow=args.dshow)
    worker.start()

    splash = V.Fonts(screen.get_height())
    screen.fill(config.GRAPE)
    V.blit_text(screen, splash, splash.huge, config.GAME_TITLE, config.SUNNY, outline=7,
                center=(screen.get_width() // 2, screen.get_height() // 2 - 60))
    V.blit_text(screen, splash, splash.mid, "Loading...",
                (255, 255, 255),
                center=(screen.get_width() // 2, screen.get_height() // 2 + 60))
    pygame.display.flip()

    audio = Audio()
    audio.init()

    game = Game(screen, worker, audio)
    try:
        game.run()
    finally:
        try:
            game.recorder.stop()
        except Exception:
            pass
        worker.stop()
        audio.shutdown()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
