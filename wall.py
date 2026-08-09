"""WALP 2 - the wall.

A slab with a hole cut in the shape of the pose. Clear it and the wall keeps
travelling, flies straight past you and bursts into spinning shards. That
fly-past is the whole point of the genre and it is what version 1 lacked.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

import config
import poses
from visuals import Color, ease_out_cubic, lerp, lerp_color

# Wall states
APPROACHING = "approaching"
PASSING = "passing"      # cleared: sails past the player
BLOCKED = "blocked"      # missed: stops dead at the line
DONE = "done"


@dataclass
class Shard:
    points: List[Tuple[float, float]]
    x: float
    y: float
    vx: float
    vy: float
    spin: float
    color: Tuple[int, int, int] = (255, 255, 255)
    angle: float = 0.0
    age: float = 0.0


@dataclass
class Wall:
    template: poses.PoseTemplate
    color: Color
    x_center_frac: float
    travel_seconds: float
    judge_at: float
    tolerance: float

    progress: float = 0.0
    state: str = APPROACHING
    result: Optional[str] = None       # "hit" | "crack" | "miss"
    final_score: float = 0.0
    state_age: float = 0.0
    shards: List[Shard] = field(default_factory=list)
    announced: bool = False
    ticked_at: float = 0.0

    # -- lifecycle -------------------------------------------------------
    @property
    def judging(self) -> bool:
        return self.state == APPROACHING and self.progress >= self.judge_at

    @property
    def finished(self) -> bool:
        return self.state == DONE

    def resolve(self, result: str, score: float) -> None:
        self.result = result
        self.final_score = score
        self.state_age = 0.0
        self.state = PASSING if result in ("hit", "crack") else BLOCKED

    def update(self, dt: float) -> None:
        self.state_age += dt
        if self.state == APPROACHING:
            self.progress += dt / max(self.travel_seconds, 1e-3)
        elif self.state == PASSING:
            # Once smashed the wall is finished almost immediately. Its debris
            # lives on separately so the next wall is already on its way -
            # waiting for the last shard to land is what made this feel slow.
            if self.state_age >= config.WALL_PASS_SECONDS:
                self.state = DONE
        elif self.state == BLOCKED:
            if self.state_age >= config.MISS_LINGER_SECONDS:
                self.state = DONE


def compute_rect(wall: Wall, play: pygame.Rect,
                 player_box: Optional[pygame.Rect],
                 x_center: Optional[float] = None) -> pygame.Rect:
    """Where the wall is right now, in screen pixels.

    Height is matched to the player (with a little slack so the hole always
    feels roomy) and grows past the screen as the wall flies by.
    """
    p = wall.progress
    far_h = config.WALL_FAR_H_FRAC * play.h
    if player_box is not None and player_box.height > 40:
        near_h = player_box.height * config.WALL_PLAYER_MARGIN
    else:
        near_h = config.WALL_FALLBACK_H_FRAC * play.h
    near_h = min(near_h, config.WALL_MAX_H_FRAC * play.h)

    if p <= 1.0:
        # Mostly linear. An eased curve keeps the wall tiny for the first
        # half of its trip, so you cannot tell what the shape is until it
        # is too late to do anything about it.
        h = lerp(far_h, near_h, p * 0.82 + ease_out_cubic(p) * 0.18)
        y_frac = lerp(config.WALL_TOP_Y_FRAC, config.JUDGMENT_LINE_Y_FRAC, p)
    else:
        # past the judgment line: rushes at the camera and drops off-screen
        over = (p - 1.0) / max(config.WALL_PASS_PROGRESS - 1.0, 1e-3)
        h = near_h * (1.0 + 1.9 * over)
        y_frac = config.JUDGMENT_LINE_Y_FRAC + 0.85 * over

    w = h * wall.template.box_aspect
    frac = wall.x_center_frac if x_center is None else x_center
    cx = play.x + frac * play.w
    cy = play.y + y_frac * play.h
    return pygame.Rect(int(cx - w / 2), int(cy - h / 2), int(w), int(h))


def _hole_polygon_points(template: poses.PoseTemplate, w: int, h: int):
    """The set of shapes that get punched out of the wall."""
    def to_local(idx: int) -> Tuple[float, float]:
        nx, ny = template.joints[idx]
        return (nx * w, ny * h)
    return to_local


def _figure_pass(surface: pygame.Surface, template: poses.PoseTemplate,
                 w: int, h: int, color, grow: int = 0) -> None:
    """Stamp the pose figure onto the surface, optionally fattened by `grow`."""
    to_local = _hole_polygon_points(template, w, h)
    # pixels per body unit, so every pose gets human proportions
    ppu = h / max(template.unit_h, 1e-3)
    limb = max(int(config.LIMB_BODY_UNITS * ppu), 10) + grow * 2
    joint_r = limb // 2
    head_r = max(int(config.HEAD_BODY_UNITS * ppu), 8) + grow

    torso = [to_local(poses.L_SHOULDER), to_local(poses.R_SHOULDER),
             to_local(poses.R_HIP), to_local(poses.L_HIP)]
    torso_i = [(int(x), int(y)) for x, y in torso]
    pygame.draw.polygon(surface, color, torso_i)
    if grow > 0:
        pygame.draw.polygon(surface, color, torso_i, grow * 2)
    # round off the torso corners so the hole is a friendly blob, not a box
    for p in torso_i:
        pygame.draw.circle(surface, color, p, joint_r)

    for a_idx, b_idx in poses.LIMB_BONES:
        pa, pb = to_local(a_idx), to_local(b_idx)
        pa_i, pb_i = (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1]))
        pygame.draw.line(surface, color, pa_i, pb_i, limb)
        pygame.draw.circle(surface, color, pa_i, joint_r)
        pygame.draw.circle(surface, color, pb_i, joint_r)

    nx, ny = to_local(poses.NOSE)
    pygame.draw.circle(surface, color, (int(nx), int(ny)), head_r)


def _smooth_figure_alpha(template: poses.PoseTemplate, w: int, h: int) -> np.ndarray:
    """Coverage map of the pose figure, blurred so it reads as one shape.

    Stamping a capsule per limb plus a disc per joint leaves a bumpy, lumpy
    outline - most obvious on a squat, where the bent knees and hips pile
    several discs on top of each other. Rendering the figure, shrinking it
    hard and blowing it back up merges all those bumps into one smooth
    silhouette, which is then thresholded at three levels to give the hole,
    its dark rim and its outer glow.

    Returns a (w, h) uint8 coverage array in pygame's x-major order.
    """
    stamp = pygame.Surface((w, h), pygame.SRCALPHA)
    _figure_pass(stamp, template, w, h, (255, 255, 255, 255), grow=0)

    # Just enough blur to round off where the joint discs meet, and no
    # more - shrink too far and the arms melt into the torso.
    small_h = 150
    factor = max(2, int(h / small_h))
    sw, sh = max(4, w // factor), max(4, h // factor)
    blurred = pygame.transform.smoothscale(stamp, (sw, sh))
    blurred = pygame.transform.smoothscale(blurred, (w, h))
    return np.array(pygame.surfarray.pixels_alpha(blurred), dtype=np.uint8)


def carve_hole(surface: pygame.Surface, template: poses.PoseTemplate,
               w: int, h: int, accent: Tuple[int, int, int]) -> None:
    """Punch the pose-shaped hole with a dark rim and a bright outer glow, so
    it reads as a real opening rather than a paler patch of wall."""
    if w < 12 or h < 12:
        return
    coverage = _smooth_figure_alpha(template, w, h)
    glow = tuple(min(255, int(c * 0.35 + 190)) for c in accent)

    rgb = pygame.surfarray.pixels3d(surface)
    alpha = pygame.surfarray.pixels_alpha(surface)
    try:
        # One blurred field, three cuts: outer glow, dark rim, then the hole.
        rgb[coverage > 70] = glow
        rgb[coverage > 118] = config.INK
        alpha[coverage > 150] = 0
    finally:
        del rgb
        del alpha


class WallRenderer:
    """Draws the wall. Caches the carved body so we only rebuild it when the
    wall changes size enough to matter."""

    def __init__(self) -> None:
        self._cache_key = None
        self._cache: Optional[pygame.Surface] = None

    def _body(self, wall: Wall, w: int, h: int) -> pygame.Surface:
        key = (wall.template.name, wall.color, w // 24, h // 24)
        if key == self._cache_key and self._cache is not None:
            # plain scale, not smoothscale: this only ever stretches by a few
            # pixels and smoothscale on a wall-sized surface is not cheap
            return pygame.transform.scale(self._cache, (w, h))

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        radius = max(6, int(min(w, h) * config.WALL_CORNER_RADIUS_FRAC))
        dark = tuple(max(0, int(c * 0.72)) for c in wall.color)
        light = tuple(min(255, int(c * 1.0 + 40)) for c in wall.color)

        pygame.draw.rect(surf, wall.color, (0, 0, w, h), border_radius=radius)
        # top highlight band gives the flat rectangle some body
        band = pygame.Surface((w, int(h * 0.34)), pygame.SRCALPHA)
        pygame.draw.rect(band, (*light, 90), band.get_rect(),
                         border_radius=radius)
        surf.blit(band, (0, 0))
        # brick-ish seams. Opaque on purpose: pygame.draw writes RGBA straight
        # into an SRCALPHA surface without blending, so a translucent colour
        # here would punch translucent stripes through the wall.
        seam = dark
        rows = max(3, h // 90)
        for r in range(1, rows):
            y = int(h * r / rows)
            pygame.draw.line(surf, seam, (6, y), (w - 6, y), max(2, h // 220))

        carve_hole(surf, wall.template, w, h, wall.color)
        pygame.draw.rect(surf, config.INK, (0, 0, w, h),
                         width=max(3, int(h * 0.014)), border_radius=radius)

        try:
            surf = surf.convert_alpha()
        except pygame.error:
            pass
        self._cache_key = key
        self._cache = surf
        return surf

    def draw(self, surface: pygame.Surface, wall: Wall, rect: pygame.Rect,
             phase: float, score: Optional[float]) -> None:
        w, h = rect.width, rect.height
        if w <= 6 or h <= 6:
            return

        if wall.state == PASSING:
            return          # it burst; the loose shards are what you see now

        body = self._body(wall, w, h)

        if wall.state == BLOCKED:
            # squash on impact
            t = min(1.0, wall.state_age / 0.22)
            sq = 1.0 + 0.16 * math.sin(t * math.pi)
            body = pygame.transform.smoothscale(
                body, (max(1, int(w * sq)), max(1, int(h / sq))))
            rect = body.get_rect(center=rect.center)

        surface.blit(body, rect.topleft)

        if wall.judging and wall.state == APPROACHING:
            t = (math.sin(phase * 9.0) + 1) * 0.5
            col = lerp_color(config.SUNNY, config.CORAL, t)
            pygame.draw.rect(surface, col, rect.inflate(18, 18),
                             width=max(3, int(6 * (0.5 + t))),
                             border_radius=max(8, int(min(w, h) * 0.1)))


def make_shards(wall: Wall, rect: pygame.Rect,
                count: int = config.SHARD_COUNT) -> List[Shard]:
    """Break the wall into chunky spinning pieces flying away from the hole."""
    cx, cy = rect.center
    rng = random.Random()
    shards: List[Shard] = []
    for _ in range(count):
        # pick a point somewhere on the wall, biased to the edges
        edge = rng.random()
        if edge < 0.5:
            px = rng.choice([rect.left, rect.right]) + rng.uniform(-40, 40)
            py = rng.uniform(rect.top, rect.bottom)
        else:
            px = rng.uniform(rect.left, rect.right)
            py = rng.choice([rect.top, rect.bottom]) + rng.uniform(-40, 40)

        size = rng.uniform(rect.height * 0.06, rect.height * 0.16)
        sides = rng.randint(3, 5)
        pts = []
        for i in range(sides):
            ang = 2 * math.pi * i / sides + rng.uniform(-0.35, 0.35)
            r = size * rng.uniform(0.55, 1.0)
            pts.append((math.cos(ang) * r, math.sin(ang) * r))

        dx, dy = px - cx, py - cy
        dist = math.hypot(dx, dy) or 1.0
        speed = rng.uniform(320, 900)
        shards.append(Shard(
            points=pts, x=px, y=py,
            vx=dx / dist * speed + rng.uniform(-120, 120),
            vy=dy / dist * speed - rng.uniform(180, 520),
            spin=rng.uniform(-config.SHARD_SPIN_MAX, config.SHARD_SPIN_MAX),
            color=wall.color,
        ))
    return shards


def update_shards(shards: List[Shard], dt: float) -> List[Shard]:
    """Debris outlives the wall it came from, so it is stepped separately."""
    alive = []
    for s in shards:
        s.age += dt
        if s.age >= config.SHARD_LIFETIME:
            continue
        s.vy += config.SHARD_GRAVITY * dt
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.angle += s.spin * dt
        alive.append(s)
    return alive


def draw_shards(surface: pygame.Surface, shards: List[Shard]) -> None:
    for s in shards:
        fade = 1.0 - s.age / config.SHARD_LIFETIME
        ca, sa = math.cos(s.angle), math.sin(s.angle)
        pts = [(s.x + px * ca - py * sa, s.y + px * sa + py * ca)
               for px, py in s.points]
        col = tuple(int(c * (0.55 + 0.45 * fade)) for c in s.color)
        pygame.draw.polygon(surface, col, pts)
        pygame.draw.polygon(surface, config.INK, pts, 3)


_GHOST_CACHE: Dict[tuple, pygame.Surface] = {}


def draw_ghost_target(surface: pygame.Surface, template: poses.PoseTemplate,
                      box: pygame.Rect, color: Color, alpha: int = 90,
                      width_scale: float = 1.0) -> None:
    """A faint 'stand like this' figure drawn over the player.

    The wall shows the shape at a distance; this shows it at the player's own
    size and position, which is a much easier thing for a small child to copy.
    Cached by size bucket - the player's box wobbles every frame and rebuilding
    this surface each time was pure waste.
    """
    w, h = box.size
    if w < 8 or h < 8:
        return
    key = (template.name, color, alpha, round(width_scale, 2), w // 16, h // 16)
    ghost = _GHOST_CACHE.get(key)
    if ghost is None:
        if len(_GHOST_CACHE) > 40:
            _GHOST_CACHE.clear()
        ghost = pygame.Surface((w, h), pygame.SRCALPHA)
        thick = max(4, int(h * 0.055 * width_scale))

        def local(idx):
            nx, ny = template.joints[idx]
            return (int(nx * w), int(ny * h))

        for a, b in poses.BONES:
            pygame.draw.line(ghost, (*color, alpha), local(a), local(b), thick)
        for idx in poses.USED_LANDMARKS:
            pygame.draw.circle(ghost, (*color, alpha), local(idx), thick // 2)
        head_r = max(6, int(h * config.WALL_HEAD_RADIUS_FRAC))
        pygame.draw.circle(ghost, (*color, alpha), local(poses.NOSE), head_r,
                           max(3, thick // 2))
        try:
            ghost = ghost.convert_alpha()
        except pygame.error:
            pass
        _GHOST_CACHE[key] = ghost
    if ghost.get_size() != (w, h):
        ghost = pygame.transform.scale(ghost, (w, h))
    surface.blit(ghost, box.topleft)


def draw_pose_guide(surface: pygame.Surface, template: poses.PoseTemplate,
                    box: pygame.Rect, bone_colors: Optional[Dict] = None,
                    base: Color = (255, 255, 255),
                    outline: Color = config.INK) -> None:
    """'Stand like this' reference figure.

    Same drawing as the icon, but each limb is tinted by how well the player
    is currently matching it - so the panel answers both "what shape?" and
    "which arm is wrong?" at a glance.
    """
    w, h = box.size
    if w < 10 or h < 10:
        return
    thick = max(3, int(h * 0.085))

    def local(idx):
        nx, ny = template.joints[idx]
        return (int(box.x + nx * w), int(box.y + ny * h))

    for a, b in poses.BONES:
        pygame.draw.line(surface, outline, local(a), local(b), thick + 5)
    head_r = max(5, int(h * config.WALL_HEAD_RADIUS_FRAC * 1.05))
    pygame.draw.circle(surface, outline, local(poses.NOSE), head_r + 3)

    for bone in poses.BONES:
        a, b = bone
        col = (bone_colors or {}).get(bone, base)
        pygame.draw.line(surface, col, local(a), local(b), thick)
    for idx in poses.USED_LANDMARKS:
        pygame.draw.circle(surface, base, local(idx), max(2, thick // 2))
    pygame.draw.circle(surface, base, local(poses.NOSE), head_r)


def draw_pose_icon(surface: pygame.Surface, template: poses.PoseTemplate,
                   box: pygame.Rect, color: Color = (255, 255, 255),
                   outline: Color = config.INK) -> None:
    """Small stick figure used wherever the pose name appears, so pre-readers
    have something to go on."""
    w, h = box.size
    if w < 10 or h < 10:
        return
    thick = max(3, int(h * 0.075))

    def local(idx):
        nx, ny = template.joints[idx]
        return (int(box.x + nx * w), int(box.y + ny * h))

    for a, b in poses.BONES:
        pygame.draw.line(surface, outline, local(a), local(b), thick + 4)
    head_r = max(5, int(h * config.WALL_HEAD_RADIUS_FRAC * 1.05))
    pygame.draw.circle(surface, outline, local(poses.NOSE), head_r + 2)
    for a, b in poses.BONES:
        pygame.draw.line(surface, color, local(a), local(b), thick)
    pygame.draw.circle(surface, color, local(poses.NOSE), head_r)
