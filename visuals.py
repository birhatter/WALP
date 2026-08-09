"""WALP 2 - everything you look at.

Fonts, themed worlds, particles, HUD widgets, the camera view and the
touchless hand-cursor UI. Anything expensive is pre-rendered once and cached; the
per-frame path is blits only.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

import config

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------
# Small maths helpers
# --------------------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    return (int(lerp(c1[0], c2[0], t)), int(lerp(c1[1], c2[1], t)),
            int(lerp(c1[2], c2[2], t)))


def score_to_color(score: float) -> Color:
    score = max(0.0, min(1.0, score))
    if score <= 0.5:
        return lerp_color(config.SCORE_BAD, config.SCORE_OK, score / 0.5)
    return lerp_color(config.SCORE_OK, config.SCORE_GOOD, (score - 0.5) / 0.5)


def ease_out_back(t: float) -> float:
    t = max(0.0, min(1.0, t))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def hue_color(h: float) -> Color:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.75, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
# Empty on purpose: fall through to pygame's built-in bold face, which is
# the type the original WALP used and the one that was asked for here.
# Put a font name in this list to override it.
FONT_PREFERENCES = []


class Fonts:
    """Heavy, clean type at sizes that scale with the screen."""

    def __init__(self, screen_h: int) -> None:
        s = screen_h / config.DESIGN_H
        name = self._pick()
        self.name = name

        def mk(px: int, bold: bool = True) -> pygame.font.Font:
            size = max(10, int(px * s))
            if name:
                return pygame.font.SysFont(name, size, bold=bold)
            return pygame.font.SysFont(None, int(size * 1.15), bold=bold)

        self.huge = mk(132)
        self.big = mk(84)
        self.title = mk(64)
        self.label = mk(56)
        self.mid = mk(42)
        self.small = mk(32)
        self.tiny = mk(24)
        self._cache: Dict[tuple, pygame.Surface] = {}

    @staticmethod
    def _pick() -> Optional[str]:
        try:
            available = set(pygame.font.get_fonts())
        except Exception:
            return None
        for pref in FONT_PREFERENCES:
            key = pref.lower().replace(" ", "")
            if key in available:
                return pref
        return None

    def render(self, font: pygame.font.Font, text: str, color: Color,
               outline: int = 4, outline_color: Color = config.INK) -> pygame.Surface:
        """Text with a thick dark outline so it stays readable over anything."""
        key = (id(font), text, color, outline, outline_color)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if len(self._cache) > 400:
            self._cache.clear()

        base = font.render(text, True, color)
        if outline <= 0:
            self._cache[key] = base
            return base

        ol = font.render(text, True, outline_color)
        w, h = base.get_size()
        surf = pygame.Surface((w + outline * 2, h + outline * 2), pygame.SRCALPHA)
        steps = 12
        for i in range(steps):
            ang = 2 * math.pi * i / steps
            surf.blit(ol, (outline + math.cos(ang) * outline,
                           outline + math.sin(ang) * outline))
        surf.blit(base, (outline, outline))
        self._cache[key] = surf
        return surf


def blit_text(surface, fonts: Fonts, font, text: str, color: Color,
              outline: int = 4, **anchor) -> pygame.Rect:
    surf = fonts.render(font, text, color, outline)
    rect = surf.get_rect(**anchor)
    surface.blit(surf, rect)
    return rect


def blit_text_scaled(surface, fonts: Fonts, font, text: str, color: Color,
                     scale: float, outline: int = 4, alpha: int = 255,
                     **anchor) -> None:
    surf = fonts.render(font, text, color, outline)
    if abs(scale - 1.0) > 0.01:
        w, h = surf.get_size()
        surf = pygame.transform.smoothscale(
            surf, (max(1, int(w * scale)), max(1, int(h * scale))))
    if alpha < 255:
        surf = surf.copy()
        surf.set_alpha(alpha)
    surface.blit(surf, surf.get_rect(**anchor))


# --------------------------------------------------------------------------
# Worlds: gradient sky + parallax hills + drifting motifs
# --------------------------------------------------------------------------

def _fast(surf: pygame.Surface, alpha: bool) -> pygame.Surface:
    """Match the display format. Skipping this makes every blit convert the
    surface on the fly, which was costing more than all the drawing combined."""
    try:
        return surf.convert_alpha() if alpha else surf.convert()
    except pygame.error:
        return surf


def make_gradient(w: int, h: int, top: Color, bottom: Color) -> pygame.Surface:
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = (np.array(top, dtype=np.float32)[None, :] * (1 - ramp)
           + np.array(bottom, dtype=np.float32)[None, :] * ramp)
    arr = np.repeat(col[:, None, :], w, axis=1).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(arr.transpose(1, 0, 2)))
    return _fast(surf, alpha=False)


def make_hill_band(w: int, h: int, color: Color, bumps: int, seed: int) -> pygame.Surface:
    """One rounded hill silhouette, tileable left-to-right."""
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    rng = random.Random(seed)
    base_y = int(h * 0.55)
    pygame.draw.rect(surf, color, (0, base_y, w, h - base_y))
    step = w / max(bumps, 1)
    for i in range(bumps + 1):
        cx = int(i * step)
        r = int(step * rng.uniform(0.45, 0.85))
        cy = base_y + rng.randint(-int(h * 0.06), int(h * 0.04))
        pygame.draw.circle(surf, color, (cx, cy), r)
    return _fast(surf, alpha=True)


def make_motif_sprite(kind: str, size: int, color: Color) -> pygame.Surface:
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    if kind == "cloud":
        white = (255, 255, 255, 205)
        pygame.draw.circle(s, white, (c - size // 5, c + size // 10), size // 4)
        pygame.draw.circle(s, white, (c + size // 5, c + size // 10), size // 4)
        pygame.draw.circle(s, white, (c, c - size // 12), size // 3)
        pygame.draw.rect(s, white, (c - size // 3, c + size // 12, size * 2 // 3, size // 5))
    elif kind == "leaf":
        col = (*color, 190)
        pygame.draw.ellipse(s, col, (size * 0.15, size * 0.3, size * 0.7, size * 0.4))
        pygame.draw.line(s, (255, 255, 255, 120), (size * 0.2, size * 0.5),
                         (size * 0.8, size * 0.5), max(1, size // 16))
    elif kind == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = c * (1.0 if i % 2 == 0 else 0.42)
            pts.append((c + math.cos(ang) * r, c + math.sin(ang) * r))
        pygame.draw.polygon(s, (255, 255, 255, 215), pts)
    elif kind == "bubble":
        pygame.draw.circle(s, (255, 255, 255, 70), (c, c), c - 1)
        pygame.draw.circle(s, (255, 255, 255, 190), (c, c), c - 1, max(1, size // 12))
        pygame.draw.circle(s, (255, 255, 255, 220), (int(c * 0.65), int(c * 0.6)),
                           max(1, size // 8))
    elif kind == "candy":
        pygame.draw.circle(s, (*color, 220), (c, c), c - 2)
        pygame.draw.arc(s, (255, 255, 255, 220), (2, 2, size - 4, size - 4),
                        0.4, 2.2, max(2, size // 7))
    else:  # snow
        pygame.draw.circle(s, (255, 255, 255, 210), (c, c), max(2, c // 2))
        for i in range(6):
            ang = i * math.pi / 3
            pygame.draw.line(s, (255, 255, 255, 150), (c, c),
                             (c + math.cos(ang) * c, c + math.sin(ang) * c),
                             max(1, size // 12))
    return _fast(s, alpha=True)


@dataclass
class MotifParticle:
    x: float
    y: float
    vx: float
    vy: float
    size_idx: int
    spin: float = 0.0
    phase: float = 0.0


class World:
    """A themed backdrop. Everything heavy is built once, on construction."""

    MOTIF_SIZES = (26, 40, 58, 82)

    def __init__(self, index: int, w: int, h: int) -> None:
        self.index = index % len(config.WORLDS)
        self.theme = config.WORLDS[self.index]
        self.w, self.h = w, h
        self.sky = make_gradient(w, h, self.theme["sky_top"], self.theme["sky_bottom"])

        band_h = int(h * 0.42)
        self.bands: List[Tuple[pygame.Surface, float, int]] = []
        for i, col in enumerate(self.theme["hills"]):
            band = make_hill_band(w + 200, band_h, col, bumps=5 + i * 3, seed=index * 7 + i)
            y = int(h - band_h * (1.0 - i * 0.10)) + int(h * 0.10)
            self.bands.append((band, 12.0 + i * 22.0, y))
        self.offsets = [0.0 for _ in self.bands]

        scale = h / config.DESIGN_H
        self.sprites = [
            make_motif_sprite(self.theme["motif"], max(8, int(sz * scale)),
                              self.theme["accent"])
            for sz in self.MOTIF_SIZES
        ]
        self.particles: List[MotifParticle] = []
        rng = random.Random(index * 31 + 5)
        # Kept modest: every one of these is a blit per frame on the same
        # thread the game logic runs on.
        for _ in range(14):
            self.particles.append(MotifParticle(
                x=rng.uniform(0, w), y=rng.uniform(0, h * 0.85),
                vx=rng.uniform(-26, 26), vy=rng.uniform(-14, 18),
                size_idx=rng.randrange(len(self.sprites)),
                phase=rng.uniform(0, 6.28),
            ))

    @property
    def name(self) -> str:
        return self.theme["name"]

    @property
    def accent(self) -> Color:
        return self.theme["accent"]

    def update(self, dt: float) -> None:
        for i, (_, speed, _) in enumerate(self.bands):
            self.offsets[i] = (self.offsets[i] + speed * dt) % (self.w + 200)
        for p in self.particles:
            p.phase += dt
            p.x += (p.vx + math.sin(p.phase * 0.7) * 8) * dt
            p.y += p.vy * dt
            if p.x < -100:
                p.x = self.w + 100
            elif p.x > self.w + 100:
                p.x = -100
            if p.y < -100:
                p.y = self.h * 0.9
            elif p.y > self.h * 0.95:
                p.y = -80

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.sky, (0, 0))
        for p in self.particles:
            spr = self.sprites[p.size_idx]
            surface.blit(spr, (int(p.x - spr.get_width() / 2),
                               int(p.y - spr.get_height() / 2)))
        for i, (band, _, y) in enumerate(self.bands):
            off = -self.offsets[i]
            surface.blit(band, (off, y))
            second = off + self.w + 200
            if second < self.w:          # the wrap copy is usually off-screen
                surface.blit(band, (second, y))


# --------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------

@dataclass
class Confetti:
    x: float
    y: float
    vx: float
    vy: float
    color: Color
    size: int
    spin: float
    angle: float
    age: float = 0.0


class ConfettiSystem:
    LIFETIME = 1.5
    GRAVITY = 900.0

    def __init__(self, scale: float = 1.0) -> None:
        self.items: List[Confetti] = []
        self.scale = scale
        self._sprites: Dict[Tuple[Color, int], pygame.Surface] = {}

    def _sprite(self, color: Color, size: int) -> pygame.Surface:
        key = (color, size)
        hit = self._sprites.get(key)
        if hit is None:
            hit = pygame.Surface((size, int(size * 0.62)), pygame.SRCALPHA)
            pygame.draw.rect(hit, color, hit.get_rect(), border_radius=max(1, size // 4))
            hit = _fast(hit, alpha=True)
            self._sprites[key] = hit
        return hit

    def burst(self, origin: Tuple[float, float], base: Color, count: int = 46,
              spread: float = 1.0) -> None:
        ox, oy = origin
        palette = [base, (255, 255, 255), config.SUNNY, config.MINT, config.CORAL]
        for _ in range(count):
            ang = random.uniform(-math.pi * 0.95, -math.pi * 0.05)
            speed = random.uniform(320, 880) * spread * self.scale
            self.items.append(Confetti(
                x=ox, y=oy,
                vx=math.cos(ang) * speed,
                vy=math.sin(ang) * speed,
                color=random.choice(palette),
                size=max(5, int(random.uniform(11, 22) * self.scale)),
                spin=random.uniform(-12, 12),
                angle=random.uniform(0, 6.28),
            ))

    def update_and_draw(self, surface: pygame.Surface, dt: float) -> None:
        alive: List[Confetti] = []
        for p in self.items:
            p.age += dt
            if p.age >= self.LIFETIME:
                continue
            p.vy += self.GRAVITY * self.scale * dt
            p.vx *= 0.99
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            spr = self._sprite(p.color, p.size)
            rot = pygame.transform.rotate(spr, math.degrees(p.angle))
            fade = 1.0 - p.age / self.LIFETIME
            if fade < 0.4:
                rot = rot.copy()
                rot.set_alpha(int(255 * fade / 0.4))
            surface.blit(rot, rot.get_rect(center=(int(p.x), int(p.y))))
            alive.append(p)
        self.items = alive

    def clear(self) -> None:
        self.items.clear()


@dataclass
class Sparkle:
    x: float
    y: float
    vx: float
    vy: float
    color: Color
    size: float
    age: float = 0.0
    life: float = 0.6


class SparkleSystem:
    def __init__(self) -> None:
        self.items: List[Sparkle] = []

    def add(self, x: float, y: float, color: Color, count: int = 1,
            speed: float = 120.0, size: float = 9.0) -> None:
        for _ in range(count):
            ang = random.uniform(0, 6.28)
            self.items.append(Sparkle(
                x=x, y=y,
                vx=math.cos(ang) * random.uniform(0.2, 1.0) * speed,
                vy=math.sin(ang) * random.uniform(0.2, 1.0) * speed,
                color=color, size=size * random.uniform(0.6, 1.3),
                life=random.uniform(0.4, 0.8),
            ))

    def update_and_draw(self, surface: pygame.Surface, dt: float) -> None:
        alive = []
        for s in self.items:
            s.age += dt
            if s.age >= s.life:
                continue
            s.x += s.vx * dt
            s.y += s.vy * dt
            s.vy += 200 * dt
            t = 1.0 - s.age / s.life
            r = max(1, int(s.size * t))
            spot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(spot, (*s.color, int(235 * t)), (r, r), r)
            surface.blit(spot, (int(s.x - r), int(s.y - r)))
            alive.append(s)
        self.items = alive

    def clear(self) -> None:
        self.items.clear()


# --------------------------------------------------------------------------
# HUD widgets
# --------------------------------------------------------------------------

def draw_panel(surface: pygame.Surface, rect: pygame.Rect, color=(20, 18, 36),
               alpha: int = 214, radius: int = 24, border: Color = (86, 80, 124),
               border_px: int = 3) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)
    surface.blit(panel, rect.topleft)
    if border_px:
        pygame.draw.rect(surface, border, rect, width=border_px, border_radius=radius)


def draw_heart(surface: pygame.Surface, center: Tuple[int, int], size: int,
               color: Color, alpha: int = 255) -> None:
    s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
    r = size * 0.31
    pygame.draw.circle(s, (*color, alpha), (int(size * 0.68), int(size * 0.72)), int(r))
    pygame.draw.circle(s, (*color, alpha), (int(size * 1.32), int(size * 0.72)), int(r))
    pygame.draw.polygon(s, (*color, alpha), [
        (size * 0.33, size * 0.86), (size * 1.67, size * 0.86), (size, size * 1.72)])
    surface.blit(s, (center[0] - size, center[1] - size))


def draw_hearts(surface: pygame.Surface, lives: int, max_lives: int, topleft,
                size: int, break_index: Optional[int] = None,
                break_age: Optional[float] = None,
                gain_index: Optional[int] = None,
                gain_age: Optional[float] = None) -> None:
    x, y = topleft
    spacing = int(size * 2.1)
    for i in range(max_lives):
        cx = x + i * spacing + size
        cy = y + size
        color = config.HEART_FULL if i < lives else config.HEART_EMPTY
        scale = 1.0
        shake = 0
        if break_index == i and break_age is not None and break_age < 0.5:
            t = break_age / 0.5
            shake = int(12 * math.sin(t * 40) * (1 - t))
            color = lerp_color(config.HEART_FULL, config.HEART_EMPTY, t)
        if gain_index == i and gain_age is not None and gain_age < 0.6:
            scale = 1.0 + 0.7 * math.sin(math.pi * (gain_age / 0.6))
        draw_heart(surface, (cx + shake, cy), max(4, int(size * scale)),
                   config.INK, 255)
        draw_heart(surface, (cx + shake, cy), max(3, int(size * scale * 0.82)),
                   color, 255)


def draw_star(surface: pygame.Surface, center: Tuple[int, int], radius: float,
              color: Color, outline: Color = config.INK) -> None:
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((center[0] + math.cos(ang) * r, center[1] + math.sin(ang) * r))
    pygame.draw.polygon(surface, outline, pts)
    pts2 = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = (radius - 4) if i % 2 == 0 else (radius - 4) * 0.45
        pts2.append((center[0] + math.cos(ang) * r, center[1] + math.sin(ang) * r))
    pygame.draw.polygon(surface, color, pts2)


def draw_progress_ring(surface: pygame.Surface, center: Tuple[int, int],
                       radius: int, frac: float, color: Color,
                       thickness: int, track: Optional[Color] = (70, 62, 96)) -> None:
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    if track is not None:
        pygame.draw.circle(surface, track, center, radius, thickness)
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.005:
        return
    # pygame.draw.arc measures angles counter-clockwise from 3 o'clock with y
    # flipped, so +pi/2 is the TOP of the circle (the opposite of what the
    # cos/sin screen maths elsewhere in this file uses). Filling clockwise
    # from 12 o'clock therefore means sweeping *down* from +pi/2.
    top = math.pi / 2
    pygame.draw.arc(surface, color, rect, top - 2 * math.pi * frac, top, thickness)


class ScoreGauge:
    """The big friendly face+ring that tells the kid how close they are.

    This ring shows exactly the number that decides pass/fail. The little
    notch on the ring is the pass mark, so 'fill past the notch' is the
    whole rule of the game, explained without words.
    """

    def __init__(self, radius: int, panel: bool = True) -> None:
        self.radius = radius
        self.panel = panel

    def draw(self, surface: pygame.Surface, fonts: Fonts, score: Optional[float],
             phase: float, center: Tuple[int, int]) -> None:
        cx, cy = center
        r = self.radius
        thick = max(8, int(r * 0.26))

        if self.panel:
            pad = int(r * 0.42)
            box = pygame.Rect(0, 0, (r + thick * 2 + pad) * 2,
                              (r + thick * 2 + pad) * 2 + int(r * 0.30))
            box.center = (cx, cy + int(r * 0.15))
            draw_panel(surface, box, color=(16, 14, 30), alpha=200,
                       radius=int(r * 0.34), border=(92, 86, 130), border_px=3)

        if score is None:
            pygame.draw.circle(surface, config.INK, center, r + 6)
            pygame.draw.circle(surface, (58, 54, 82), center, r)
            blit_text(surface, fonts, fonts.mid, "--", (150, 146, 172),
                      outline=0, center=(cx, cy))
            return

        score = max(0.0, min(1.0, score))
        color = score_to_color(score)
        passed = score >= config.PASS_THRESHOLD
        pulse = 1.0 + (0.05 * math.sin(phase * 8.0) if passed else 0.0)

        ring_r = int(r + thick * 1.7)
        draw_progress_ring(surface, center, ring_r, score, color, thick)

        # The pass mark. "Fill the ring past the marker" is the whole rule.
        ang = -math.pi / 2 + 2 * math.pi * config.PASS_THRESHOLD
        tip_r = ring_r - thick * 0.62
        out_r = ring_r + thick * 1.15
        nose = (cx + math.cos(ang) * tip_r, cy + math.sin(ang) * tip_r)
        side = ang + math.pi / 2
        spread = thick * 0.5
        wing1 = (cx + math.cos(ang) * out_r + math.cos(side) * spread,
                 cy + math.sin(ang) * out_r + math.sin(side) * spread)
        wing2 = (cx + math.cos(ang) * out_r - math.cos(side) * spread,
                 cy + math.sin(ang) * out_r - math.sin(side) * spread)
        pygame.draw.polygon(surface, config.INK, [nose, wing1, wing2])
        pygame.draw.polygon(surface, (255, 255, 255), [nose, wing1, wing2], 3)

        rr = max(4, int(r * pulse))
        pygame.draw.circle(surface, config.INK, center, rr + 5)
        pygame.draw.circle(surface, (28, 26, 44), center, rr)
        pygame.draw.circle(surface, color, center, rr, max(3, int(rr * 0.10)))

        # A live percentage says far more than a smiley face, and it is the
        # actual number the pass/fail test uses. Sized off the ring so it
        # cannot outgrow the circle it sits in.
        num = fonts.render(fonts.big, f"{int(score * 100)}", color, 0)
        max_w = int(rr * 1.45)
        if num.get_width() > max_w:
            k = max_w / num.get_width()
            num = pygame.transform.smoothscale(
                num, (max_w, max(1, int(num.get_height() * k))))
        surface.blit(num, num.get_rect(center=(cx, cy - int(rr * 0.13))))
        blit_text(surface, fonts, fonts.tiny, "MATCH", (176, 174, 204),
                  outline=0, center=(cx, cy + int(rr * 0.52)))


# --------------------------------------------------------------------------
# Camera view
# --------------------------------------------------------------------------

class CameraView:
    """Live picture-in-picture of the player.

    Display only - the frame is converted, drawn and dropped. Nothing here
    touches the disk. The image is mirrored so it behaves like a mirror.
    """

    def __init__(self) -> None:
        self._surf: Optional[pygame.Surface] = None
        self._frame_id = -1
        self._mask: Optional[pygame.Surface] = None
        self._mask_size = None

    def _rounded_mask(self, size: Tuple[int, int], radius: int) -> pygame.Surface:
        if self._mask is None or self._mask_size != (size, radius):
            m = pygame.Surface(size, pygame.SRCALPHA)
            pygame.draw.rect(m, (255, 255, 255, 255), m.get_rect(),
                             border_radius=radius)
            self._mask = _fast(m, alpha=True)
            self._mask_size = (size, radius)
        return self._mask

    def draw(self, surface: pygame.Surface, preview, frame_id: int,
             rect: pygame.Rect, accent: Color, label: str = "") -> None:
        if preview is None:
            return
        radius = max(6, int(rect.height * 0.12))
        if frame_id != self._frame_id or self._surf is None                 or self._surf.get_size() != rect.size:
            try:
                arr = np.ascontiguousarray(preview, dtype=np.uint8)
                h, w = arr.shape[0], arr.shape[1]
                img = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGB")
                img = pygame.transform.flip(img, True, False)   # mirror
                img = pygame.transform.smoothscale(img, rect.size)
                img = img.convert_alpha()
                img.blit(self._rounded_mask(rect.size, radius), (0, 0),
                         special_flags=pygame.BLEND_RGBA_MULT)
                self._surf = img
                self._frame_id = frame_id
            except (ValueError, pygame.error):
                return

        shadow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 110), shadow.get_rect(),
                         border_radius=radius)
        surface.blit(shadow, (rect.x + 5, rect.y + 6))
        surface.blit(self._surf, rect.topleft)
        pygame.draw.rect(surface, config.INK, rect.inflate(8, 8),
                         width=max(3, int(rect.height * 0.035)),
                         border_radius=radius + 4)
        pygame.draw.rect(surface, accent, rect.inflate(1, 1),
                         width=max(2, int(rect.height * 0.018)),
                         border_radius=radius)
        if label:
            tag = pygame.Rect(rect.x, rect.bottom - int(rect.height * 0.20),
                              rect.width, int(rect.height * 0.20))
            strip = pygame.Surface(tag.size, pygame.SRCALPHA)
            strip.fill((*config.INK, 170))
            surface.blit(strip, tag.topleft)


# --------------------------------------------------------------------------
# Touchless hand-cursor UI
# --------------------------------------------------------------------------

class HandCursor:
    """Maps the player's raised hand to a screen pointer."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.pos = [screen_w * 0.5, screen_h * 0.5]
        self.active = False
        self.radius = int(screen_h * config.CURSOR_RADIUS_FRAC)
        self.trail: List[Tuple[float, float]] = []

    def update(self, point: Optional[Tuple[float, float]], dt: float) -> None:
        if point is None:
            # Hold the pointer for a moment after losing the hand. Tracking
            # drops a frame here and there, especially from across a room,
            # and cancelling a half-made selection every time is maddening.
            self._grace = max(0.0, getattr(self, "_grace", 0.0) - dt)
            self.active = self._grace > 0.0
            return
        self._grace = config.CURSOR_GRACE_SECONDS
        self.active = True
        a = min(1.0, config.CURSOR_SMOOTHING * dt * 60 / 8.0 + 0.12)
        self.pos[0] += (point[0] - self.pos[0]) * a
        self.pos[1] += (point[1] - self.pos[1]) * a
        self.trail.append((self.pos[0], self.pos[1]))
        del self.trail[:-9]

    def draw(self, surface: pygame.Surface, color: Color) -> None:
        if not self.active:
            return
        for i, (x, y) in enumerate(self.trail):
            t = (i + 1) / len(self.trail)
            r = max(2, int(self.radius * t * 0.8))
            spot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(spot, (*color, int(70 * t)), (r, r), r)
            surface.blit(spot, (int(x - r), int(y - r)))
        p = (int(self.pos[0]), int(self.pos[1]))
        pygame.draw.circle(surface, config.INK, p, self.radius + 4)
        pygame.draw.circle(surface, (255, 255, 255), p, self.radius)
        pygame.draw.circle(surface, color, p, max(2, int(self.radius * 0.55)))


class HoverButton:
    """A big round target. Hold the hand over it to press - no clicking."""

    def __init__(self, key: str, label: str, center: Tuple[int, int], radius: int,
                 color: Color, icon: str = "") -> None:
        self.key = key
        self.label = label
        self.center = center
        self.radius = radius
        self.color = color
        self.icon = icon
        self.hover = 0.0
        self.pop = 0.0
        self.enabled = True

    def contains(self, pos: Sequence[float]) -> bool:
        return math.hypot(pos[0] - self.center[0], pos[1] - self.center[1]) <= self.radius * 1.15

    def update(self, cursor: HandCursor, dt: float, focused: bool = None) -> bool:
        """Returns True on the frame it fires.

        `focused` lets the caller nominate this button as the selected one
        (nearest to the pointer) instead of requiring the pointer to land
        inside it, which needs far less precision.
        """
        if not self.enabled:
            self.hover = 0.0
            return False
        inside = focused if focused is not None else (
            cursor.active and self.contains(cursor.pos))
        if inside:
            was = self.hover
            self.hover = min(config.HOVER_SECONDS, self.hover + dt)
            if was < config.HOVER_SECONDS <= self.hover:
                self.pop = 0.35
                return True
        else:
            self.hover = max(0.0, self.hover - dt * 2.4)
        self.pop = max(0.0, self.pop - dt)
        return False

    @property
    def just_entered(self) -> bool:
        return 0.0 < self.hover < 0.12

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        frac = self.hover / config.HOVER_SECONDS
        scale = 1.0 + 0.09 * frac + (0.22 * self.pop / 0.35 if self.pop else 0.0)
        r = int(self.radius * scale)
        cx, cy = self.center

        shadow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(shadow, (0, 0, 0, 70), (r, r), r)
        surface.blit(shadow, (cx - r, cy - r + int(r * 0.12)))

        body = self.color if self.enabled else (140, 136, 158)
        pygame.draw.circle(surface, config.INK, (cx, cy), r + 5)
        pygame.draw.circle(surface, body, (cx, cy), r)
        gloss = pygame.Surface((int(r * 1.3), int(r * 0.9)), pygame.SRCALPHA)
        pygame.draw.ellipse(gloss, (255, 255, 255, 70), gloss.get_rect())
        surface.blit(gloss, (cx - int(r * 0.65), cy - int(r * 0.72)))

        if frac > 0.01:
            draw_progress_ring(surface, (cx, cy), r + 12, frac, config.SUNNY,
                               max(5, int(r * 0.16)), track=config.INK)

        if self.icon:
            blit_text(surface, fonts, fonts.title, self.icon, (255, 255, 255),
                      outline=3, center=(cx, cy))
        blit_text(surface, fonts, fonts.small, self.label, (255, 255, 255),
                  outline=4, midtop=(cx, cy + r + 14))


# --------------------------------------------------------------------------
# Player rendering
# --------------------------------------------------------------------------

def draw_player_glow(surface: pygame.Surface, points: Dict[int, Tuple[int, int]],
                     bones: Sequence[Tuple[int, int]],
                     bone_colors: Dict[Tuple[int, int], Color],
                     skin: Color, head: Optional[Tuple[Tuple[int, int], int]],
                     thickness: int, scratch: Optional[pygame.Surface] = None) -> None:
    """Neon body outline. Each limb is tinted by how well that limb matches,
    so a child can see which arm to move without reading a word.

    Draws into a reusable scratch surface and only touches the player's
    bounding box - clearing and blitting the whole screen every frame was the
    single most expensive thing the renderer did.
    """
    if not points:
        return
    glow = scratch if scratch is not None else pygame.Surface(
        surface.get_size(), pygame.SRCALPHA)

    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    pad = int(thickness * 3) + 8
    if head is not None:
        xs.append(head[0][0]); ys.append(head[0][1] - head[1])
        pad = max(pad, head[1] + thickness * 2)
    area = pygame.Rect(min(xs) - pad, min(ys) - pad,
                       max(xs) - min(xs) + pad * 2, max(ys) - min(ys) + pad * 2)
    area = area.clip(glow.get_rect())
    if area.width <= 0 or area.height <= 0:
        return
    glow.fill((0, 0, 0, 0), area)

    passes = ((thickness * 2.2, 34), (thickness * 1.5, 80), (thickness, 150),
              (thickness * 0.45, 255))
    for width, alpha in passes:
        w = max(2, int(width))
        for bone in bones:
            a, b = bone
            if a in points and b in points:
                col = bone_colors.get(bone, skin)
                pygame.draw.line(glow, (*col, alpha), points[a], points[b], w)
        for p in points.values():
            pygame.draw.circle(glow, (*skin, alpha), p, w // 2 + 2)
        if head is not None:
            centre, radius = head
            pygame.draw.circle(glow, (*skin, alpha), centre, radius, max(2, w // 2))
    surface.blit(glow, area.topleft, area=area, special_flags=pygame.BLEND_RGBA_ADD)


_MASK_CACHE = {"key": None, "surf": None}


def mask_to_surface(mask: np.ndarray, size: Tuple[int, int], color: Color,
                    alpha: int = 110, frame_id: int = -1
                    ) -> Optional[pygame.Surface]:
    """Turn MediaPipe's segmentation mask into a soft coloured silhouette.

    Cached against the camera frame it came from. Scaling a mask up to the
    full window is not cheap, and the camera only produces ~25 new frames a
    second while this is asked for 60 times a second - so five out of every
    eight rebuilds were pure waste.
    """
    if mask is None:
        return None
    key = (frame_id, size, color, alpha)
    if frame_id >= 0 and _MASK_CACHE["key"] == key:
        return _MASK_CACHE["surf"]
    try:
        m = np.clip(mask, 0.0, 1.0)
        h, w = m.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = (m * alpha).astype(np.uint8)
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA")
        surf = pygame.transform.flip(surf, True, False)   # play field is mirrored
        surf = pygame.transform.smoothscale(surf, size)
        if frame_id >= 0:
            _MASK_CACHE["key"] = key
            _MASK_CACHE["surf"] = surf
        return surf
    except (ValueError, pygame.error):
        return None


# --------------------------------------------------------------------------
# Transitions & screen effects
# --------------------------------------------------------------------------

class Transition:
    """Circular wipe between screens - stops mode changes feeling like a jump cut."""

    def __init__(self) -> None:
        self.t = 1.0
        self.duration = 0.45
        self.pending = None
        self.color = config.GRAPE

    def start(self, callback, color: Color = config.GRAPE) -> None:
        if self.pending is None:
            self.t = 0.0
            self.pending = callback
            self.color = color

    @property
    def busy(self) -> bool:
        return self.t < 1.0

    def update(self, dt: float) -> None:
        if self.t >= 1.0:
            return
        was = self.t
        self.t = min(1.0, self.t + dt / self.duration)
        if was < 0.5 <= self.t and self.pending is not None:
            cb, self.pending = self.pending, None
            cb()

    def draw(self, surface: pygame.Surface) -> None:
        if self.t >= 1.0:
            return
        w, h = surface.get_size()
        full = math.hypot(w, h) * 0.55
        if self.t < 0.5:
            r = full * (1.0 - ease_out_cubic(self.t * 2))
        else:
            r = full * ease_out_cubic((self.t - 0.5) * 2)
        cover = pygame.Surface((w, h), pygame.SRCALPHA)
        cover.fill((*self.color, 255))
        pygame.draw.circle(cover, (0, 0, 0, 0), (w // 2, h // 2), max(0, int(r)))
        surface.blit(cover, (0, 0))


def draw_vignette_flash(surface: pygame.Surface, color: Color, strength: float) -> None:
    if strength <= 0.01:
        return
    w, h = surface.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((*color, int(90 * min(1.0, strength))))
    surface.blit(overlay, (0, 0))


class Shake:
    def __init__(self) -> None:
        self.t = 0.0
        self.magnitude = 0.0

    def kick(self, magnitude: float, duration: float = 0.35) -> None:
        self.magnitude = max(self.magnitude, magnitude)
        self.t = max(self.t, duration)
        self.duration = duration

    def update(self, dt: float) -> Tuple[int, int]:
        if self.t <= 0:
            return (0, 0)
        self.t = max(0.0, self.t - dt)
        k = self.t / max(getattr(self, "duration", 0.35), 1e-3)
        m = self.magnitude * k
        return (int(math.sin(self.t * 61) * m), int(math.cos(self.t * 47) * m))
