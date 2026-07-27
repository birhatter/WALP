import colorsys
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import pygame

FRAME_W, FRAME_H = 1280, 720
FPS = 30

PIP_W, PIP_H = 320, 180
PIP_MARGIN = 24

VISIBILITY_THRESHOLD = 0.3

WALL_TRAVEL_SECONDS = 4.5
JUDGMENT_WINDOW_START = 0.78
JUDGMENT_PROGRESS = 0.90
LINGER_FRAMES = 20
SPAWN_DELAY_FRAMES = 45

SCORE_PER_HIT = 100
FLASH_DURATION = 18
SHAKE_DURATION = 12
SHAKE_MAGNITUDE = 14

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

BG_COLOR = (10, 10, 25)
NEON_PLAYER_COLOR = (0, 255, 255)
JUDGMENT_LINE_COLOR = (255, 255, 255)
JUDGMENT_LINE_Y_FRAC = 0.72

MIN_WALL_H_FRAC, MAX_WALL_H_FRAC = 0.14, 0.82
TOP_Y_FRAC = 0.08
WALL_X_CENTER_FRAC = 0.5

MATCH_SCORE_ZERO_ERROR_MULT = 1.5
STATUS_ALMOST_COLOR = (255, 200, 60)
STATUS_GREAT_THRESHOLD = 0.80
STATUS_ALMOST_THRESHOLD = 0.45
STATUS_PANEL_RADIUS_FRAC = 0.125
STATUS_PANEL_MARGIN_PX = 90
STATUS_RING_THICKNESS = 16
STATUS_PULSE_SPEED = 5.5
STATUS_PULSE_AMPLITUDE = 0.05
STATUS_SPARKLE_COUNT = 6
STATUS_NEUTRAL_COLOR = (95, 95, 108)
LIVE_SCORE_EMA_ALPHA = 0.3

HIT_LINGER_FRAMES = 60
CONFETTI_COUNT_PER_HIT = 40
CONFETTI_GRAVITY = 600.0
CONFETTI_SPEED_MIN, CONFETTI_SPEED_MAX = 250.0, 650.0
CONFETTI_LIFETIME = 1.1
CONFETTI_RADIUS_MIN, CONFETTI_RADIUS_MAX = 4, 9
CELEBRATION_TEXTS = ("YOU DID IT!", 'SUPER!', "AWESOME!")
CELEBRATION_TEXT_DURATION = 1.0
SCORE_POP_DURATION = 0.35
SCORE_POP_SCALE = 1.5
FLOATING_SCORE_RISE_PX = 90
FLOATING_SCORE_DURATION = 1.0
COMBO_BADGE_BURST_DURATION = 0.4
LABEL_BOUNCE_DURATION = 0.4

FLASH_ALPHA_MAX = 70
MISS_MESSAGES = ('TRY AGAIN!', "SO CLOSE!")
MISS_MESSAGE_DURATION = 0.9
HEART_SIZE_PX = 26
HEART_SPACING_PX = 54
HEART_BREAK_DURATION = 0.5

COUNTDOWN_BAR_HEIGHT_PX = 22
COUNTDOWN_BAR_MARGIN_PX = 40
COUNTDOWN_BAR_Y_PX = 100
COUNTDOWN_BAR_BG_COLOR = (40, 40, 55)
COUNTDOWN_SAFE_COLOR = (90, 200, 255)
COUNTDOWN_WARN_COLOR = (255, 210, 60)
COUNTDOWN_DANGER_COLOR = (255, 80, 80)
COUNTDOWN_BLINK_SPEED = 8.0

WALL_GLOW_PULSE_SPEED = 6.0
WALL_GLOW_MAX_WIDTH = 10

DIFFICULTY_HITS_PER_STEP = 3
DIFFICULTY_STEP_SECONDS = 0.3
WALL_TRAVEL_SECONDS_FLOOR = 3.0

FILLED_SILHOUETTE_ALPHA = 70

MODE_START = "start"
MODE_PLAYING = "playing"
MODE_GAME_OVER = "game_over"
POSE_HOLD_SECONDS = 1.0
GAME_TITLE = "WALP"

BEST_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_score.txt")

STAR_RATING_THRESHOLDS = (3, 8, 15)
STAR_POP_INTERVAL = 0.4
STAR_POP_SCALE_DURATION = 0.3


NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

USED_LANDMARKS = [
    NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
]

BONES = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
]

MIRROR_INDEX = {
    NOSE: NOSE,
    L_SHOULDER: R_SHOULDER, R_SHOULDER: L_SHOULDER,
    L_ELBOW: R_ELBOW, R_ELBOW: L_ELBOW,
    L_WRIST: R_WRIST, R_WRIST: L_WRIST,
    L_HIP: R_HIP, R_HIP: L_HIP,
    L_KNEE: R_KNEE, R_KNEE: L_KNEE,
    L_ANKLE: R_ANKLE, R_ANKLE: L_ANKLE,
}

JOINT_ANGLE_TRIPLES = [
    ("sol_dirsek", L_SHOULDER, L_ELBOW, L_WRIST),
    ("sag_dirsek", R_SHOULDER, R_ELBOW, R_WRIST),
    ("sol_omuz_kaldirma", L_HIP, L_SHOULDER, L_ELBOW),
    ('sag_omuz_kaldirma', R_HIP, R_SHOULDER, R_ELBOW),
    ("sol_govde_bacak", L_SHOULDER, L_HIP, L_KNEE),
    ('sag_govde_bacak', R_SHOULDER, R_HIP, R_KNEE),
    ("sol_diz", L_HIP, L_KNEE, L_ANKLE),
    ("sag_diz", R_HIP, R_KNEE, R_ANKLE),
]

def joint_angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(abs(cross), dot))


def lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t

def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def score_to_color(score: float) -> Tuple[int, int, int]:
    score = max(0.0, min(1.0, score))
    if score <= 0.5:
        return lerp_color(MATCH_BAD_COLOR, STATUS_ALMOST_COLOR, score / 0.5)
    return lerp_color(STATUS_ALMOST_COLOR, MATCH_OK_COLOR, (score - 0.5) / 0.5)


JointMap = Dict[int, Tuple[float, float]]

T_POSE_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.66, 0.28), R_SHOULDER: (0.34, 0.28),
    L_ELBOW: (0.86, 0.28), R_ELBOW: (0.14, 0.28),
    L_WRIST: (0.97, 0.28), R_WRIST: (0.03, 0.28),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

Y_POSE_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.78, 0.22), R_ELBOW: (0.22, 0.22),
    L_WRIST: (0.90, 0.15), R_WRIST: (0.10, 0.15),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

ONE_ARM_UP_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.72, 0.22), R_ELBOW: (0.34, 0.44),
    L_WRIST: (0.80, 0.15), R_WRIST: (0.32, 0.58),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

X_POSE_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.64, 0.26), R_SHOULDER: (0.36, 0.26),
    L_ELBOW: (0.82, 0.16), R_ELBOW: (0.18, 0.16),
    L_WRIST: (0.96, 0.15), R_WRIST: (0.04, 0.15),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.62, 0.72), R_KNEE: (0.38, 0.72),
    L_ANKLE: (0.70, 0.90), R_ANKLE: (0.30, 0.90),
}

L_POSE_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.62, 0.20), R_ELBOW: (0.14, 0.30),
    L_WRIST: (0.62, 0.15), R_WRIST: (0.02, 0.30),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

SQUAT_JOINTS: JointMap = {
    NOSE: (0.50, 0.30),
    L_SHOULDER: (0.62, 0.46), R_SHOULDER: (0.38, 0.46),
    L_ELBOW: (0.72, 0.56), R_ELBOW: (0.28, 0.56),
    L_WRIST: (0.78, 0.64), R_WRIST: (0.22, 0.64),
    L_HIP: (0.62, 0.64), R_HIP: (0.38, 0.64),
    L_KNEE: (0.70, 0.80), R_KNEE: (0.30, 0.80),
    L_ANKLE: (0.58, 0.90), R_ANKLE: (0.42, 0.90),
}

HANDS_ON_HEAD_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.82, 0.22), R_ELBOW: (0.18, 0.22),
    L_WRIST: (0.54, 0.15), R_WRIST: (0.46, 0.15),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

CHEER_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.60, 0.18), R_ELBOW: (0.40, 0.18),
    L_WRIST: (0.58, 0.15), R_WRIST: (0.42, 0.15),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.44, 0.72),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.46, 0.90),
}

KICK_JOINTS: JointMap = {
    NOSE: (0.50, 0.15),
    L_SHOULDER: (0.62, 0.30), R_SHOULDER: (0.38, 0.30),
    L_ELBOW: (0.74, 0.40), R_ELBOW: (0.26, 0.40),
    L_WRIST: (0.82, 0.48), R_WRIST: (0.18, 0.48),
    L_HIP: (0.58, 0.52), R_HIP: (0.42, 0.52),
    L_KNEE: (0.56, 0.72), R_KNEE: (0.48, 0.46),
    L_ANKLE: (0.54, 0.90), R_ANKLE: (0.54, 0.60),
}

SQUAT_REQUIRED_TRIPLES = ("sol_govde_bacak", "sag_govde_bacak", 'sol_diz', "sag_diz")

KICK_REQUIRED_TRIPLES = SQUAT_REQUIRED_TRIPLES

HANDS_ON_HEAD_REQUIRED_TRIPLES = ("sol_dirsek", 'sag_dirsek', "sol_omuz_kaldirma", "sag_omuz_kaldirma")

CHEER_REQUIRED_TRIPLES = ("sol_omuz_kaldirma", "sag_omuz_kaldirma")

L_POSE_REQUIRED_TRIPLES = ("sol_omuz_kaldirma", "sag_omuz_kaldirma")

X_POSE_REQUIRED_TRIPLES = ("sol_omuz_kaldirma", "sag_omuz_kaldirma", "sol_govde_bacak", 'sag_govde_bacak')

@dataclass(frozen=True)
class JointAngleSpec:
    name: str
    a_idx: int
    b_idx: int
    c_idx: int
    target_deg: float
    tolerance_deg: float = 35.0


@dataclass(frozen=True)
class PoseTemplate:
    name: str
    display_label: str
    joints: JointMap
    angle_specs: Tuple[JointAngleSpec, ...]
    box_aspect: float
    required_triples: Tuple[str, ...]

def _scaled(joints: JointMap, idx: int) -> Tuple[float, float]:
    x, y = joints[idx]
    return (x * FRAME_W, y * FRAME_H)


def build_pose_template(
    name: str,
    display_label: str,
    joints: JointMap,
    box_aspect: float,
    required_triples: Optional[Tuple[str, ...]] = None,
    tolerance_deg: float = 35.0,
) -> PoseTemplate:
    specs = []
    for spec_name, a, b, c in JOINT_ANGLE_TRIPLES:
        target = joint_angle_deg(_scaled(joints, a), _scaled(joints, b), _scaled(joints, c))
        specs.append(JointAngleSpec(spec_name, a, b, c, target, tolerance_deg))
    if required_triples is None:
        required_triples = tuple(name for name, *_ in JOINT_ANGLE_TRIPLES)
    return PoseTemplate(name, display_label, joints, tuple(specs), box_aspect, required_triples)

POSE_TEMPLATES: List[PoseTemplate] = [
    build_pose_template('T_POSE', "T-POSE", T_POSE_JOINTS, box_aspect=0.95),
    build_pose_template("Y_POSE", "Y-POSE", Y_POSE_JOINTS, box_aspect=0.80),
    build_pose_template("ONE_ARM_UP", "ONE ARM UP", ONE_ARM_UP_JOINTS, box_aspect=0.62),
    build_pose_template(
        "X_POSE", "X-POSE", X_POSE_JOINTS, box_aspect=1.05,
        required_triples=X_POSE_REQUIRED_TRIPLES, tolerance_deg=42.0,
    ),
    build_pose_template(
        "L_POSE", "L-POSE", L_POSE_JOINTS, box_aspect=0.68,
        required_triples=L_POSE_REQUIRED_TRIPLES, tolerance_deg=38.0,
    ),
    build_pose_template(
        "SQUAT", 'SQUAT', SQUAT_JOINTS, box_aspect=0.75,
        required_triples=SQUAT_REQUIRED_TRIPLES, tolerance_deg=45.0,
    ),
    build_pose_template(
        "HANDS_ON_HEAD", 'HANDS ON HEAD', HANDS_ON_HEAD_JOINTS, box_aspect=0.85,
        required_triples=HANDS_ON_HEAD_REQUIRED_TRIPLES, tolerance_deg=40.0,
    ),
    build_pose_template(
        "CHEER", "CHEER!", CHEER_JOINTS, box_aspect=0.55,
        required_triples=CHEER_REQUIRED_TRIPLES, tolerance_deg=38.0,
    ),
    build_pose_template(
        "KICK", "KICK", KICK_JOINTS, box_aspect=0.72,
        required_triples=KICK_REQUIRED_TRIPLES, tolerance_deg=42.0,
    ),
]

POSE_TEMPLATES_BY_NAME: Dict[str, PoseTemplate] = {t.name: t for t in POSE_TEMPLATES}
T_POSE_TEMPLATE = POSE_TEMPLATES_BY_NAME["T_POSE"]

def mirror_angle_specs(specs: Tuple[JointAngleSpec, ...]) -> Tuple[JointAngleSpec, ...]:
    return tuple(
        JointAngleSpec(
            s.name,
            MIRROR_INDEX[s.a_idx], MIRROR_INDEX[s.b_idx], MIRROR_INDEX[s.c_idx],
            s.target_deg, s.tolerance_deg,
        )
        for s in specs
    )


def angle_for_triple(landmarks, a_idx: int, b_idx: int, c_idx: int) -> Optional[float]:
    for idx in (a_idx, b_idx, c_idx):
        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < VISIBILITY_THRESHOLD:
            return None
    a = (landmarks[a_idx].x * FRAME_W, landmarks[a_idx].y * FRAME_H)
    b = (landmarks[b_idx].x * FRAME_W, landmarks[b_idx].y * FRAME_H)
    c = (landmarks[c_idx].x * FRAME_W, landmarks[c_idx].y * FRAME_H)
    return joint_angle_deg(a, b, c)


def _specs_match(specs: Tuple[JointAngleSpec, ...], required_triples: Tuple[str, ...], landmarks) -> bool:
    for spec in specs:
        if spec.name not in required_triples:
            continue
        angle = angle_for_triple(landmarks, spec.a_idx, spec.b_idx, spec.c_idx)
        if angle is None or abs(angle - spec.target_deg) > spec.tolerance_deg:
            return False
    return True

def check_pose_match(template: PoseTemplate, landmarks) -> bool:
    if _specs_match(template.angle_specs, template.required_triples, landmarks):
        return True
    return _specs_match(mirror_angle_specs(template.angle_specs), template.required_triples, landmarks)

def _score_specs(specs: Tuple[JointAngleSpec, ...], landmarks) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for spec in specs:
        angle = angle_for_triple(landmarks, spec.a_idx, spec.b_idx, spec.c_idx)
        if angle is None:
            scores[spec.name] = 0.0
            continue
        zero_at = spec.tolerance_deg * MATCH_SCORE_ZERO_ERROR_MULT
        error = abs(angle - spec.target_deg)
        scores[spec.name] = max(0.0, 1.0 - error / zero_at)
    return scores


def graded_pose_score(template: PoseTemplate, landmarks) -> Tuple[float, Dict[int, float]]:

    def joint_scores_from(specs: Tuple[JointAngleSpec, ...], by_name: Dict[str, float]) -> Dict[int, float]:
        return {spec.b_idx: by_name[spec.name] for spec in specs}

    name_scores_o = _score_specs(template.angle_specs, landmarks)
    mirrored_specs = mirror_angle_specs(template.angle_specs)
    name_scores_m = _score_specs(mirrored_specs, landmarks)

    required = template.required_triples
    avg_o = sum(name_scores_o[n] for n in required) / len(required)
    avg_m = sum(name_scores_m[n] for n in required) / len(required)

    if avg_o >= avg_m:
        return avg_o, joint_scores_from(template.angle_specs, name_scores_o)
    return avg_m, joint_scores_from(mirrored_specs, name_scores_m)


@dataclass
class Wall:
    template: PoseTemplate
    color: Tuple[int, int, int]
    x_center_frac: float
    progress: float = 0.0
    resolved: bool = False
    result: Optional[str] = None
    linger: int = 0
    final_score: Optional[float] = None


def random_neon_color() -> Tuple[int, int, int]:
    hue = random.random()
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class ConfettiParticle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    radius: float
    age: float = 0.0


def spawn_wall() -> Wall:
    template = random.choice(POSE_TEMPLATES)
    return Wall(template=template, color=random_neon_color(), x_center_frac=WALL_X_CENTER_FRAC)

LIMB_BONES = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

LIMB_THICKNESS_FRAC = 0.20
HEAD_RADIUS_FRAC = 0.10

LIMB_BONE_VERTEX = {
    (L_SHOULDER, L_ELBOW): L_SHOULDER, (L_ELBOW, L_WRIST): L_ELBOW,
    (R_SHOULDER, R_ELBOW): R_SHOULDER, (R_ELBOW, R_WRIST): R_ELBOW,
    (L_HIP, L_KNEE): L_HIP, (L_KNEE, L_ANKLE): L_KNEE,
    (R_HIP, R_KNEE): R_HIP, (R_KNEE, R_ANKLE): R_KNEE,
}
LIMB_SCORE_EMA_ALPHA = 0.25


def current_wall_travel_seconds(walls_cleared: int) -> float:
    steps = walls_cleared // DIFFICULTY_HITS_PER_STEP
    return max(WALL_TRAVEL_SECONDS_FLOOR, WALL_TRAVEL_SECONDS - steps * DIFFICULTY_STEP_SECONDS)

def compute_wall_rect(wall: Wall, play_rect: pygame.Rect, player_bbox: Optional[pygame.Rect]) -> pygame.Rect:
    progress = min(wall.progress, 1.0)
    far_h = MIN_WALL_H_FRAC * play_rect.h
    near_h = player_bbox.height if player_bbox is not None else MAX_WALL_H_FRAC * play_rect.h
    wall_h = int(lerp(far_h, near_h, progress))
    wall_w = int(wall_h * wall.template.box_aspect)
    center_x = play_rect.x + wall.x_center_frac * play_rect.w
    center_y = play_rect.y + lerp(TOP_Y_FRAC, JUDGMENT_LINE_Y_FRAC, progress) * play_rect.h
    return pygame.Rect(int(center_x - wall_w / 2), int(center_y - wall_h / 2), wall_w, wall_h)

def render_wall(screen: pygame.Surface, wall: Wall, wall_rect: pygame.Rect) -> None:
    wall_w, wall_h = wall_rect.width, wall_rect.height
    if wall_w <= 4 or wall_h <= 4:
        return

    wall_surf = pygame.Surface((wall_w, wall_h), pygame.SRCALPHA)
    wall_surf.fill((*wall.color, 255))

    def to_local(idx: int) -> Tuple[float, float]:
        nx, ny = wall.template.joints[idx]
        return (nx * wall_w, ny * wall_h)

    limb_thickness = max(int(wall_h * LIMB_THICKNESS_FRAC), 14)
    joint_radius = limb_thickness // 2
    head_radius = max(int(wall_h * HEAD_RADIUS_FRAC), 16)

    TRANSPARENT = (0, 0, 0, 0)

    torso_poly = [to_local(L_SHOULDER), to_local(R_SHOULDER), to_local(R_HIP), to_local(L_HIP)]
    pygame.draw.polygon(wall_surf, TRANSPARENT, [(int(x), int(y)) for x, y in torso_poly])

    for a_idx, b_idx in LIMB_BONES:
        pa, pb = to_local(a_idx), to_local(b_idx)
        pygame.draw.line(wall_surf, TRANSPARENT, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), limb_thickness)
        for t in (0.0, 0.33, 0.66, 1.0):
            x = pa[0] + (pb[0] - pa[0]) * t
            y = pa[1] + (pb[1] - pa[1]) * t
            pygame.draw.circle(wall_surf, TRANSPARENT, (int(x), int(y)), joint_radius)

    nose_x, nose_y = to_local(NOSE)
    pygame.draw.circle(wall_surf, TRANSPARENT, (int(nose_x), int(nose_y)), head_radius)

    screen.blit(wall_surf, wall_rect.topleft)
    pygame.draw.rect(screen, tuple(min(c + 60, 255) for c in wall.color), wall_rect, width=3)

def draw_wall_judgment_glow(screen: pygame.Surface, wall: Wall, wall_rect: pygame.Rect, phase: float) -> None:
    if wall.resolved or wall.progress < JUDGMENT_WINDOW_START:
        return
    t = (math.sin(phase * WALL_GLOW_PULSE_SPEED) + 1.0) * 0.5
    width = max(2, int(lerp(2, WALL_GLOW_MAX_WIDTH, t)))
    color = lerp_color(COUNTDOWN_WARN_COLOR, COUNTDOWN_DANGER_COLOR, t)
    pygame.draw.rect(screen, color, wall_rect.inflate(width * 2, width * 2), width=width)


def to_screen_point(lm, play_rect: pygame.Rect) -> Tuple[float, float]:
    x = play_rect.w - lm.x * play_rect.w
    y = lm.y * play_rect.h
    return (x, y)


BBOX_PADDING_FRAC = 0.12

def compute_player_bbox_rect(landmarks, play_rect: pygame.Rect) -> Optional[pygame.Rect]:
    points = []
    for idx in USED_LANDMARKS:
        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < VISIBILITY_THRESHOLD:
            continue
        points.append(to_screen_point(lm, play_rect))
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(max_x - min_x, 1.0) * BBOX_PADDING_FRAC
    pad_y = max(max_y - min_y, 1.0) * BBOX_PADDING_FRAC
    min_x, max_x = min_x - pad_x, max_x + pad_x
    min_y, max_y = min_y - pad_y, max_y + pad_y
    return pygame.Rect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))


def draw_neon_silhouette(
    screen: pygame.Surface,
    landmarks,
    play_rect: pygame.Rect,
    joint_scores: Optional[Dict[int, float]] = None,
    limb_ema: Optional[Dict[Tuple[int, int], float]] = None,
) -> None:
    glow = pygame.Surface((play_rect.w, play_rect.h), pygame.SRCALPHA)

    points: Dict[int, Tuple[int, int]] = {}
    for idx in USED_LANDMARKS:
        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < VISIBILITY_THRESHOLD:
            continue
        x, y = to_screen_point(lm, play_rect)
        points[idx] = (int(x), int(y))

    bone_colors: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for bone in BONES:
        vertex = LIMB_BONE_VERTEX.get(bone) if joint_scores is not None else None
        if vertex is None:
            bone_colors[bone] = NEON_PLAYER_COLOR
            continue
        raw = joint_scores.get(vertex, 0.0)
        if limb_ema is not None:
            prev = limb_ema.get(bone)
            raw = raw if prev is None else lerp(prev, raw, LIMB_SCORE_EMA_ALPHA)
            limb_ema[bone] = raw
        bone_colors[bone] = score_to_color(raw)

    passes = [(10, 40), (7, 90), (4, 150), (2, 255)]
    r, g, b = NEON_PLAYER_COLOR
    for width, alpha in passes:
        for a_idx, b_idx in BONES:
            if a_idx in points and b_idx in points:
                br, bg, bb = bone_colors[(a_idx, b_idx)]
                pygame.draw.line(glow, (br, bg, bb, alpha), points[a_idx], points[b_idx], width)
        for p in points.values():
            pygame.draw.circle(glow, (r, g, b, alpha), p, width + 2)

    if L_SHOULDER in points and R_SHOULDER in points and NOSE in points:
        shoulder_w = math.hypot(points[L_SHOULDER][0] - points[R_SHOULDER][0],
                                 points[L_SHOULDER][1] - points[R_SHOULDER][1])
        head_radius = max(int(shoulder_w * 0.45), 10)
        head_center = (points[NOSE][0], points[NOSE][1] - head_radius // 3)
        for width, alpha in passes:
            pygame.draw.circle(glow, (r, g, b, alpha), head_center, head_radius, max(width - 2, 1))

    screen.blit(glow, play_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)


def draw_filled_silhouette(
    cache_surf: pygame.Surface,
    landmarks,
    play_rect: pygame.Rect,
    player_bbox: Optional[pygame.Rect],
    tint: Tuple[int, int, int],
) -> None:
    cache_surf.fill((0, 0, 0, 0))

    points: Dict[int, Tuple[int, int]] = {}
    for idx in USED_LANDMARKS:
        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < VISIBILITY_THRESHOLD:
            continue
        x, y = to_screen_point(lm, play_rect)
        points[idx] = (int(x), int(y))

    color = (*tint, FILLED_SILHOUETTE_ALPHA)
    thickness = max(int(player_bbox.height * LIMB_THICKNESS_FRAC), 14) if player_bbox is not None else 30

    if L_SHOULDER in points and R_SHOULDER in points and L_HIP in points and R_HIP in points:
        torso_poly = [points[L_SHOULDER], points[R_SHOULDER], points[R_HIP], points[L_HIP]]
        pygame.draw.polygon(cache_surf, color, torso_poly)

    for a_idx, b_idx in LIMB_BONES:
        if a_idx in points and b_idx in points:
            pygame.draw.line(cache_surf, color, points[a_idx], points[b_idx], thickness)
            pygame.draw.circle(cache_surf, color, points[a_idx], thickness // 2)
            pygame.draw.circle(cache_surf, color, points[b_idx], thickness // 2)

    if L_SHOULDER in points and R_SHOULDER in points and NOSE in points:
        shoulder_w = math.hypot(points[L_SHOULDER][0] - points[R_SHOULDER][0],
                                 points[L_SHOULDER][1] - points[R_SHOULDER][1])
        head_radius = max(int(shoulder_w * 0.5), 12)
        pygame.draw.circle(cache_surf, color, points[NOSE], head_radius)


def draw_dashed_line(screen: pygame.Surface, y: int, width: int, color, dash=18, gap=14) -> None:
    x = 0
    while x < width:
        pygame.draw.line(screen, color, (x, y), (min(x + dash, width), y), 3)
        x += dash + gap


def draw_countdown_bar(screen: pygame.Surface, wall: Optional[Wall], phase: float, screen_w: int) -> None:
    if wall is None or wall.resolved:
        return
    x = COUNTDOWN_BAR_MARGIN_PX
    y = COUNTDOWN_BAR_Y_PX
    w = screen_w - COUNTDOWN_BAR_MARGIN_PX * 2
    h = COUNTDOWN_BAR_HEIGHT_PX
    pygame.draw.rect(screen, COUNTDOWN_BAR_BG_COLOR, (x, y, w, h), border_radius=h // 2)

    progress = min(wall.progress, 1.0)
    fill_frac = 1.0 - progress
    if fill_frac <= 0.0:
        return
    if progress < JUDGMENT_WINDOW_START:
        color = COUNTDOWN_SAFE_COLOR
    else:
        t = (progress - JUDGMENT_WINDOW_START) / max(1.0 - JUDGMENT_WINDOW_START, 0.001)
        blink = 0.7 + 0.3 * math.sin(phase * COUNTDOWN_BLINK_SPEED)
        color = tuple(int(c * blink) for c in lerp_color(COUNTDOWN_WARN_COLOR, COUNTDOWN_DANGER_COLOR, t))
    fill_w = max(int(w * fill_frac), h)
    pygame.draw.rect(screen, color, (x, y, fill_w, h), border_radius=h // 2)

MATCH_OK_COLOR = (90, 235, 140)
MATCH_BAD_COLOR = (255, 90, 90)

def draw_status_panel(
    screen: pygame.Surface,
    live_score: Optional[float],
    phase: float,
    center: Tuple[int, int],
    radius: int,
    font_word: pygame.font.Font,
) -> None:
    cx, cy = center

    if live_score is None:
        pygame.draw.circle(screen, tuple(max(c - 30, 0) for c in STATUS_NEUTRAL_COLOR), center, radius)
        pygame.draw.circle(screen, STATUS_NEUTRAL_COLOR, center, radius, width=6)
        return

    live_score = max(0.0, min(1.0, live_score))
    color = score_to_color(live_score)

    pulse = 1.0
    if live_score >= STATUS_GREAT_THRESHOLD:
        pulse = 1.0 + STATUS_PULSE_AMPLITUDE * math.sin(phase * STATUS_PULSE_SPEED)
    draw_radius = max(int(radius * pulse), 1)

    pygame.draw.circle(screen, (55, 55, 68), center, radius + STATUS_RING_THICKNESS, width=STATUS_RING_THICKNESS)
    if live_score > 0.01:
        ring_rect = pygame.Rect(cx - radius - STATUS_RING_THICKNESS, cy - radius - STATUS_RING_THICKNESS,
                                 (radius + STATUS_RING_THICKNESS) * 2, (radius + STATUS_RING_THICKNESS) * 2)
        start_angle = -math.pi / 2
        end_angle = start_angle + 2 * math.pi * live_score
        pygame.draw.arc(screen, color, ring_rect, start_angle, end_angle, STATUS_RING_THICKNESS)

    pygame.draw.circle(screen, color, center, draw_radius)
    pygame.draw.circle(screen, tuple(min(c + 45, 255) for c in color), center, draw_radius, width=5)

    size = draw_radius * 0.7
    if live_score >= STATUS_GREAT_THRESHOLD:
        p1 = (cx - size * 0.5, cy + size * 0.05)
        p2 = (cx - size * 0.12, cy + size * 0.42)
        p3 = (cx + size * 0.55, cy - size * 0.38)
        pygame.draw.lines(screen, (255, 255, 255), False, [p1, p2, p3], 9)
        for i in range(STATUS_SPARKLE_COUNT):
            ang = phase * 2.0 + i * (2 * math.pi / STATUS_SPARKLE_COUNT)
            sx = cx + math.cos(ang) * (radius * 1.35)
            sy = cy + math.sin(ang) * (radius * 1.35)
            spark_r = max(int(4 + 2 * math.sin(phase * 3.0 + i)), 1)
            pygame.draw.circle(screen, (255, 255, 220), (int(sx), int(sy)), spark_r)
    elif live_score < STATUS_ALMOST_THRESHOLD:
        pygame.draw.line(screen, (255, 255, 255), (cx - size * 0.45, cy - size * 0.45), (cx + size * 0.45, cy + size * 0.45), 9)
        pygame.draw.line(screen, (255, 255, 255), (cx - size * 0.45, cy + size * 0.45), (cx + size * 0.45, cy - size * 0.45), 9)
    else:
        pygame.draw.line(screen, (255, 255, 255), (cx - size * 0.4, cy), (cx + size * 0.4, cy), 10)

    word = "GREAT!" if live_score >= STATUS_GREAT_THRESHOLD else ('ALMOST!' if live_score >= STATUS_ALMOST_THRESHOLD else "TRY!")
    word_surf = font_word.render(word, True, (255, 255, 255))
    screen.blit(word_surf, word_surf.get_rect(midtop=(cx, cy + radius + STATUS_RING_THICKNESS + 14)))

STAR_COUNT = 22
STAR_COLORS = [
    (120, 170, 255), (255, 180, 230), (170, 255, 210),
    (255, 220, 140), (190, 160, 255),
]


def init_stars(screen_w: int, screen_h: int) -> List[Dict]:
    stars = []
    for _ in range(STAR_COUNT):
        stars.append({
            "x": random.uniform(0, screen_w),
            'y': random.uniform(0, screen_h),
            "r": random.uniform(6, 22),
            "dx": random.uniform(-18, 18),
            "dy": random.uniform(8, 26),
            "color": random.choice(STAR_COLORS),
            "alpha": random.randint(35, 90),
        })
    return stars


def update_and_draw_stars(surface: pygame.Surface, stars: List[Dict], dt: float, screen_w: int, screen_h: int) -> None:
    for star in stars:
        star['x'] += star["dx"] * dt
        star["y"] += star["dy"] * dt
        if star["x"] < -30:
            star["x"] = screen_w + 30
        elif star["x"] > screen_w + 30:
            star['x'] = -30
        if star["y"] < -30:
            star['y'] = screen_h + 30
        elif star["y"] > screen_h + 30:
            star["y"] = -30

        r = int(star["r"])
        spot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(spot, (*star["color"], star["alpha"]), (r, r), r)
        surface.blit(spot, (int(star['x'] - r), int(star["y"] - r)))


def spawn_confetti_burst(origin: Tuple[float, float], base_color: Tuple[int, int, int]) -> List[ConfettiParticle]:
    ox, oy = origin
    palette = [base_color, (255, 255, 255), (255, 225, 60)]
    particles = []
    for _ in range(CONFETTI_COUNT_PER_HIT):
        angle = random.uniform(-0.9 * math.pi, -0.1 * math.pi)
        speed = random.uniform(CONFETTI_SPEED_MIN, CONFETTI_SPEED_MAX)
        particles.append(ConfettiParticle(
            x=ox, y=oy,
            vx=math.cos(angle) * speed, vy=math.sin(angle) * speed,
            color=random.choice(palette),
            radius=random.uniform(CONFETTI_RADIUS_MIN, CONFETTI_RADIUS_MAX),
        ))
    return particles


def update_and_draw_confetti(surface: pygame.Surface, particles: List[ConfettiParticle], dt: float) -> List[ConfettiParticle]:
    survivors = []
    for p in particles:
        p.age += dt
        if p.age >= CONFETTI_LIFETIME:
            continue
        p.vy += CONFETTI_GRAVITY * dt
        p.x += p.vx * dt
        p.y += p.vy * dt
        alpha = max(0, int(255 * (1.0 - p.age / CONFETTI_LIFETIME)))
        r = max(int(p.radius), 1)
        spot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(spot, (*p.color, alpha), (r, r), r)
        surface.blit(spot, (int(p.x - r), int(p.y - r)))
        survivors.append(p)
    return survivors


def draw_pip(screen: pygame.Surface, camera_frame_bgr, screen_h: int) -> None:
    rgb = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (PIP_W, PIP_H), interpolation=cv2.INTER_AREA)
    pip_surface = pygame.surfarray.make_surface(resized.transpose(1, 0, 2))
    x, y = PIP_MARGIN, screen_h - PIP_H - PIP_MARGIN
    screen.blit(pip_surface, (x, y))
    pygame.draw.rect(screen, NEON_PLAYER_COLOR, (x - 3, y - 3, PIP_W + 6, PIP_H + 6), width=3)

def render_scaled_text(font: pygame.font.Font, text: str, color, scale: float) -> pygame.Surface:
    base = font.render(text, True, color)
    if abs(scale - 1.0) < 0.01:
        return base
    w, h = base.get_size()
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return pygame.transform.smoothscale(base, (new_w, new_h))


def _banner_scale_alpha(age: float, duration: float) -> Tuple[float, int]:
    t = max(0.0, min(1.0, age / duration))
    if t < 0.6:
        seg_t = t / 0.6
        scale = lerp(0.5, 1.15, seg_t) if seg_t < 0.8 else lerp(1.15, 1.0, (seg_t - 0.8) / 0.2)
    else:
        scale = 1.0
    alpha = 255 if t < 0.7 else int(255 * (1.0 - (t - 0.7) / 0.3))
    return scale, max(0, alpha)


def draw_scaling_banner(surface: pygame.Surface, text: str, age: float, duration: float,
                         center: Tuple[int, int], font: pygame.font.Font, color) -> None:
    scale, alpha = _banner_scale_alpha(age, duration)
    if alpha <= 0:
        return
    text_surf = render_scaled_text(font, text, color, scale)
    text_surf.set_alpha(alpha)
    surface.blit(text_surf, text_surf.get_rect(center=center))

def draw_floating_score(surface: pygame.Surface, amount: int, age: float,
                         origin: Tuple[int, int], font: pygame.font.Font) -> None:
    t = max(0.0, min(1.0, age / FLOATING_SCORE_DURATION))
    y_offset = -FLOATING_SCORE_RISE_PX * t
    alpha = int(255 * (1.0 - t))
    if alpha <= 0:
        return
    surf = font.render(f"+{amount}", True, (255, 225, 60))
    surf.set_alpha(alpha)
    ox, oy = origin
    surface.blit(surf, surf.get_rect(midleft=(ox, int(oy + y_offset))))


def draw_combo_badge(surface: pygame.Surface, combo: int, badge_age: Optional[float],
                      center: Tuple[int, int], font: pygame.font.Font) -> None:
    if combo < 2:
        return
    scale = 1.0
    if badge_age is not None and badge_age < COMBO_BADGE_BURST_DURATION:
        t = badge_age / COMBO_BADGE_BURST_DURATION
        scale = lerp(1.6, 1.0, t)
        for i in range(8):
            ang = i * (2 * math.pi / 8)
            burst_r = 30 + 40 * t
            sx = center[0] + math.cos(ang) * burst_r
            sy = center[1] + math.sin(ang) * burst_r
            spark_alpha = max(0, int(255 * (1.0 - t)))
            spark = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(spark, (255, 225, 60, spark_alpha), (4, 4), 4)
            surface.blit(spark, (int(sx - 4), int(sy - 4)))
    text_surf = render_scaled_text(font, f"x{combo}", (255, 225, 60), scale)
    surface.blit(text_surf, text_surf.get_rect(center=center))


def bounce_in_scale(age: Optional[float], duration: float) -> float:
    if age is None or age >= duration:
        return 1.0
    t = age / duration
    return 1.0 + 0.6 * math.sin(t * math.pi) * (1.0 - t)


def _draw_heart(surface: pygame.Surface, center: Tuple[int, int], size: int, color, alpha: int = 255) -> None:
    cx, cy = center
    r = size // 2
    heart_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
    lc, lr = (size * 0.5, size * 0.5), r * 0.62
    rc = (size * 1.5, size * 0.5)
    pygame.draw.circle(heart_surf, (*color, alpha), (int(lc[0]), int(lc[1])), int(lr))
    pygame.draw.circle(heart_surf, (*color, alpha), (int(rc[0]), int(rc[1])), int(lr))
    tip = (size, size * 1.7)
    left = (size * 0.15, size * 0.55)
    right = (size * 1.85, size * 0.55)
    pygame.draw.polygon(heart_surf, (*color, alpha), [left, right, tip])
    surface.blit(heart_surf, (cx - size, cy - size))


def draw_hearts(surface: pygame.Surface, lives: int, max_lives: int,
                 break_index: Optional[int], break_age: Optional[float], top_left: Tuple[int, int]) -> None:
    x, y = top_left
    for i in range(max_lives):
        cx = x + i * HEART_SPACING_PX + HEART_SIZE_PX
        cy = y + HEART_SIZE_PX
        shake_x = 0
        color = (255, 90, 110) if i < lives else (65, 65, 78)
        if break_index is not None and i == break_index and break_age is not None and break_age < HEART_BREAK_DURATION:
            t = break_age / HEART_BREAK_DURATION
            shake_x = int(10 * math.sin(t * 40) * (1.0 - t))
            color = lerp_color((255, 90, 110), (65, 65, 78), t)
        _draw_heart(surface, (cx + shake_x, cy), HEART_SIZE_PX, color, 255)

def draw_pose_template_silhouette(
    surface: pygame.Surface, template: PoseTemplate, rect: pygame.Rect, color, line_width: int = 10
) -> None:
    def to_local(idx: int) -> Tuple[int, int]:
        nx, ny = template.joints[idx]
        return (int(rect.x + nx * rect.w), int(rect.y + ny * rect.h))

    for a_idx, b_idx in BONES:
        pygame.draw.line(surface, color, to_local(a_idx), to_local(b_idx), line_width)
    for idx in USED_LANDMARKS:
        pygame.draw.circle(surface, color, to_local(idx), line_width // 2 + 2)
    head_radius = max(int(rect.h * HEAD_RADIUS_FRAC), 16)
    pygame.draw.circle(surface, color, to_local(NOSE), head_radius, width=max(line_width // 2, 4))


def draw_hold_gate(
    surface: pygame.Surface, cx: int, cy: int, radius: int,
    hold_timer: float, hold_target: float, label: str, font: pygame.font.Font,
) -> None:
    frac = 0.0 if hold_target <= 0 else max(0.0, min(1.0, hold_timer / hold_target))
    pygame.draw.circle(surface, (55, 55, 68), (cx, cy), radius + STATUS_RING_THICKNESS, width=STATUS_RING_THICKNESS)
    if frac > 0.01:
        ring_rect = pygame.Rect(cx - radius - STATUS_RING_THICKNESS, cy - radius - STATUS_RING_THICKNESS,
                                 (radius + STATUS_RING_THICKNESS) * 2, (radius + STATUS_RING_THICKNESS) * 2)
        start_angle = -math.pi / 2
        end_angle = start_angle + 2 * math.pi * frac
        pygame.draw.arc(surface, MATCH_OK_COLOR, ring_rect, start_angle, end_angle, STATUS_RING_THICKNESS)
    fill_color = lerp_color(STATUS_NEUTRAL_COLOR, MATCH_OK_COLOR, frac)
    pygame.draw.circle(surface, fill_color, (cx, cy), radius)
    pygame.draw.circle(surface, tuple(min(c + 45, 255) for c in fill_color), (cx, cy), radius, width=5)
    label_surf = font.render(label, True, (255, 255, 255))
    surface.blit(label_surf, label_surf.get_rect(midtop=(cx, cy + radius + STATUS_RING_THICKNESS + 14)))

def draw_start_screen(
    screen: pygame.Surface, screen_w: int, screen_h: int, hold_timer: float,
    font_title: pygame.font.Font, font_tagline: pygame.font.Font, font_gate: pygame.font.Font,
) -> None:
    title_surf = font_title.render(GAME_TITLE, True, (0, 255, 255))
    screen.blit(title_surf, title_surf.get_rect(midtop=(screen_w // 2, 30)))
    tagline_surf = font_tagline.render("Step into the shape!", True, (255, 255, 255))
    screen.blit(tagline_surf, tagline_surf.get_rect(midtop=(screen_w // 2, 30 + title_surf.get_height() + 10)))

    box_h = int(screen_h * 0.48)
    box_w = int(box_h * T_POSE_TEMPLATE.box_aspect)
    box_rect = pygame.Rect(screen_w // 2 - box_w // 2, int(screen_h * 0.28), box_w, box_h)
    draw_pose_template_silhouette(screen, T_POSE_TEMPLATE, box_rect, (0, 255, 255), line_width=12)

    gate_radius = int(screen_h * 0.09)
    gate_center = (screen_w // 2, screen_h - gate_radius - 90)
    draw_hold_gate(screen, gate_center[0], gate_center[1], gate_radius,
                    hold_timer, POSE_HOLD_SECONDS, 'Strike a T-POSE to start', font_gate)


def stars_for_performance(walls_cleared: int) -> int:
    return sum(1 for threshold in STAR_RATING_THRESHOLDS if walls_cleared >= threshold)


def _draw_single_star(surface: pygame.Surface, center: Tuple[int, int], radius: int, color) -> None:
    cx, cy = center
    points = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
    pygame.draw.polygon(surface, color, points)

def draw_star_rating(surface: pygame.Surface, stars_earned: int, center: Tuple[int, int], age: Optional[float]) -> None:
    if stars_earned <= 0 or age is None:
        return
    cx, cy = center
    spacing = 70
    start_x = cx - (stars_earned - 1) * spacing / 2
    for i in range(stars_earned):
        local_age = age - i * STAR_POP_INTERVAL
        if local_age < 0:
            continue
        scale = 1.0
        if local_age < STAR_POP_SCALE_DURATION:
            scale = lerp(1.6, 1.0, local_age / STAR_POP_SCALE_DURATION)
        sx = start_x + i * spacing
        _draw_single_star(surface, (int(sx), cy), int(32 * scale), (255, 215, 60))


def draw_game_over_screen(
    screen: pygame.Surface, screen_w: int, screen_h: int, score: int, walls_cleared: int,
    best_score: int, new_record: bool, hold_timer: float, game_over_age: Optional[float],
    font_big: pygame.font.Font, font_mid: pygame.font.Font, font_gate: pygame.font.Font,
) -> None:
    y = 60
    title_surf = font_big.render("GREAT EFFORT!", True, (255, 255, 255))
    screen.blit(title_surf, title_surf.get_rect(midtop=(screen_w // 2, y)))
    y += title_surf.get_height() + 20

    score_surf = font_mid.render(f"Final Score: {score}", True, (255, 255, 0))
    screen.blit(score_surf, score_surf.get_rect(midtop=(screen_w // 2, y)))
    y += score_surf.get_height() + 10

    cleared_surf = font_mid.render(f"Walls Cleared: {walls_cleared}", True, (200, 220, 255))
    screen.blit(cleared_surf, cleared_surf.get_rect(midtop=(screen_w // 2, y)))
    y += cleared_surf.get_height() + 10

    if new_record:
        record_surf = font_mid.render("NEW RECORD!", True, (255, 120, 200))
    else:
        record_surf = font_mid.render(f"Best: {best_score}", True, (170, 170, 190))
    screen.blit(record_surf, record_surf.get_rect(midtop=(screen_w // 2, y)))
    y += record_surf.get_height() + 50

    draw_star_rating(screen, stars_for_performance(walls_cleared), (screen_w // 2, y), game_over_age)

    gate_radius = int(screen_h * 0.09)
    gate_center = (screen_w // 2, screen_h - gate_radius - 90)
    draw_hold_gate(screen, gate_center[0], gate_center[1], gate_radius,
                    hold_timer, POSE_HOLD_SECONDS, "Stand in T-POSE to play again", font_gate)

@dataclass
class GameState:
    score: int = 0
    active_wall: Optional[Wall] = None
    spawn_cooldown: int = 0
    flash_timer: int = 0
    shake_timer: int = 0
    panel_phase: float = 0.0
    smoothed_live_score: Optional[float] = None
    limb_score_ema: Dict[Tuple[int, int], float] = field(default_factory=dict)

    combo: int = 0
    lives: int = 3
    walls_cleared: int = 0
    celebration_text: str = ""
    celebration_age: Optional[float] = None
    confetti: List[ConfettiParticle] = field(default_factory=list)
    score_pop_age: Optional[float] = None
    floating_score_age: Optional[float] = None
    floating_score_amount: int = 0
    combo_badge_age: Optional[float] = None
    miss_message: str = ""
    miss_message_age: Optional[float] = None
    heart_break_index: Optional[int] = None
    heart_break_age: Optional[float] = None

    mode: str = MODE_START
    hold_timer: float = 0.0
    new_record: bool = False
    game_over_age: Optional[float] = None
    label_bounce_age: Optional[float] = None


def load_best_score() -> int:
    try:
        with open(BEST_SCORE_PATH, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def save_best_score(value: int) -> None:
    try:
        with open(BEST_SCORE_PATH, 'w', encoding='utf-8') as f:
            f.write(str(value))
    except OSError:
        pass


def create_landmarker():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')
    if not os.path.exists(model_path):
        raise SystemExit(
            f"HATA: Model dosyasi bulunamadi: {model_path}\n"
            "'pose_landmarker_lite.task' dosyasinin hole_in_the_wall.py ile ayni "
            "klasorde oldugundan emin olun."
        )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("HATA: Kamera acilamadi. Baska bir uygulama kullaniyor olabilir.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def create_video_writer(width: int, height: int, fps: int) -> Optional[cv2.VideoWriter]:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORDINGS_DIR, f"session_{timestamp}.avi")

    for codec in ("XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Video kaydi basladi ({codec}): {path}")
            return writer
        writer.release()

    print("UYARI: Video kaydi baslatilamadi, kayit yapilmadan oyuna devam ediliyor.")
    return None


def cleanup(cap, video_writer, landmarker) -> None:
    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()
    if landmarker is not None:
        landmarker.close()
    pygame.quit()
    print('Oyun kapatildi, kaynaklar serbest birakildi.')


def main() -> None:
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_w, screen_h = screen.get_size()
    pygame.display.set_caption('Hole in the Wall')
    clock = pygame.time.Clock()

    font_score = pygame.font.SysFont(None, 64)
    font_label = pygame.font.SysFont(None, 88, bold=True)
    font_hint = pygame.font.SysFont(None, 32)
    font_status = pygame.font.SysFont(None, 60, bold=True)
    font_celebration = pygame.font.SysFont(None, 96, bold=True)
    font_combo = pygame.font.SysFont(None, 48, bold=True)
    font_floating = pygame.font.SysFont(None, 40, bold=True)
    font_title = pygame.font.SysFont(None, 140, bold=True)
    font_tagline = pygame.font.SysFont(None, 48)
    font_gate = pygame.font.SysFont(None, 36, bold=True)
    font_gameover_big = pygame.font.SysFont(None, 90, bold=True)
    font_gameover_mid = pygame.font.SysFont(None, 52)

    play_rect = pygame.Rect(0, 0, screen_w, screen_h)
    stars = init_stars(screen_w, screen_h)

    status_panel_radius = int(screen_h * STATUS_PANEL_RADIUS_FRAC)
    status_panel_center = (screen_w - STATUS_PANEL_MARGIN_PX - status_panel_radius, screen_h // 2)

    fill_cache_surface = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)

    best_score = load_best_score()

    cap = None
    video_writer = None
    landmarker = None
    state = GameState()
    frame_index = 0
    last_landmarks = None
    camera_frame_for_pip = None

    mp_drawing = mp.tasks.vision.drawing_utils
    pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

    try:
        landmarker = create_landmarker()
        cap = open_camera()
        video_writer = create_video_writer(screen_w, screen_h, FPS)

        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
            if not running:
                break

            ok, raw_frame = cap.read()
            if ok:
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_index * (1000 / FPS))
                frame_index += 1
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    last_landmarks = result.pose_landmarks[0]
                    mp_drawing.draw_landmarks(raw_frame, last_landmarks, pose_connections)
                camera_frame_for_pip = raw_frame

            player_bbox = compute_player_bbox_rect(last_landmarks, play_rect) if last_landmarks is not None else None

            live_match: Optional[bool] = None
            live_score: Optional[float] = None
            live_joint_scores: Dict[int, float] = {}
            shake_dx = shake_dy = 0

            if state.mode != MODE_PLAYING:
                hold_score = 0.0
                if last_landmarks is not None:
                    hold_score, _ = graded_pose_score(T_POSE_TEMPLATE, last_landmarks)
                state.hold_timer = state.hold_timer + dt if hold_score >= STATUS_GREAT_THRESHOLD else 0.0
                if state.hold_timer >= POSE_HOLD_SECONDS:
                    state = GameState(mode=MODE_PLAYING)
            else:
                state.panel_phase += dt
                if state.active_wall is None:
                    if state.spawn_cooldown <= 0:
                        state.active_wall = spawn_wall()
                        state.smoothed_live_score = None
                        state.limb_score_ema.clear()
                        state.label_bounce_age = 0.0
                    else:
                        state.spawn_cooldown -= 1
                else:
                    wall = state.active_wall
                    if not wall.resolved:
                        wall.progress += dt / current_wall_travel_seconds(state.walls_cleared)
                        if last_landmarks is not None:
                            raw_score, live_joint_scores = graded_pose_score(wall.template, last_landmarks)
                            if state.smoothed_live_score is None:
                                state.smoothed_live_score = raw_score
                            else:
                                state.smoothed_live_score = lerp(state.smoothed_live_score, raw_score, LIVE_SCORE_EMA_ALPHA)
                        live_score = state.smoothed_live_score
                        if wall.progress >= JUDGMENT_WINDOW_START:
                            live_match = last_landmarks is not None and check_pose_match(wall.template, last_landmarks)
                            if live_match:
                                wall.resolved = True
                                wall.result = 'hit'
                                wall.linger = HIT_LINGER_FRAMES
                                wall.final_score = 1.0
                                state.combo += 1
                                gained = SCORE_PER_HIT * state.combo
                                state.score += gained
                                state.walls_cleared += 1
                                state.score_pop_age = 0.0
                                state.floating_score_age = 0.0
                                state.floating_score_amount = gained
                                state.celebration_age = 0.0
                                state.celebration_text = random.choice(CELEBRATION_TEXTS)
                                if state.combo >= 2:
                                    state.combo_badge_age = 0.0
                                hit_wall_rect = compute_wall_rect(wall, play_rect, player_bbox)
                                state.confetti.extend(spawn_confetti_burst(hit_wall_rect.center, wall.color))
                            elif wall.progress >= 1.0:
                                wall.resolved = True
                                wall.result = "miss"
                                wall.linger = LINGER_FRAMES
                                wall.final_score = live_score if live_score is not None else 0.0
                                state.flash_timer = FLASH_DURATION
                                state.shake_timer = SHAKE_DURATION
                                state.combo = 0
                                state.miss_message = random.choice(MISS_MESSAGES)
                                state.miss_message_age = 0.0
                                if state.lives > 0:
                                    state.heart_break_index = state.lives - 1
                                    state.lives -= 1
                                    state.heart_break_age = 0.0
                                    if state.lives <= 0:
                                        state.new_record = state.score > best_score
                                        if state.new_record:
                                            best_score = state.score
                                            save_best_score(best_score)
                                        state.mode = MODE_GAME_OVER
                                        state.hold_timer = 0.0
                                        state.game_over_age = 0.0
                    else:
                        live_score = wall.final_score
                        wall.linger -= 1
                        if wall.linger <= 0:
                            state.active_wall = None
                            state.spawn_cooldown = SPAWN_DELAY_FRAMES

                if state.celebration_age is not None:
                    state.celebration_age += dt
                if state.score_pop_age is not None:
                    state.score_pop_age += dt
                if state.floating_score_age is not None:
                    state.floating_score_age += dt
                if state.combo_badge_age is not None:
                    state.combo_badge_age += dt
                if state.miss_message_age is not None:
                    state.miss_message_age += dt
                if state.heart_break_age is not None:
                    state.heart_break_age += dt
                if state.label_bounce_age is not None:
                    state.label_bounce_age += dt

                if state.shake_timer > 0:
                    t = state.shake_timer
                    shake_dx = int(SHAKE_MAGNITUDE * math.sin(t * 2.5) * (t / SHAKE_DURATION))
                    shake_dy = int(SHAKE_MAGNITUDE * math.cos(t * 1.7) * (t / SHAKE_DURATION))
                    state.shake_timer -= 1

            if state.mode == MODE_GAME_OVER and state.game_over_age is not None:
                state.game_over_age += dt

            play_surface = pygame.Surface((screen_w, screen_h))
            play_surface.fill(BG_COLOR)
            update_and_draw_stars(play_surface, stars, dt, screen_w, screen_h)

            if state.mode == MODE_PLAYING:
                judgment_y = int(JUDGMENT_LINE_Y_FRAC * play_rect.h)
                draw_dashed_line(play_surface, judgment_y, screen_w, JUDGMENT_LINE_COLOR)

                if state.active_wall is not None:
                    wall_rect = compute_wall_rect(state.active_wall, play_rect, player_bbox)
                    render_wall(play_surface, state.active_wall, wall_rect)
                    draw_wall_judgment_glow(play_surface, state.active_wall, wall_rect, state.panel_phase)

                if last_landmarks is not None:
                    if state.active_wall is not None:
                        tint = score_to_color(live_score if live_score is not None else 0.0)
                        draw_filled_silhouette(fill_cache_surface, last_landmarks, play_rect, player_bbox, tint)
                        play_surface.blit(fill_cache_surface, (0, 0))
                    draw_neon_silhouette(
                        play_surface, last_landmarks, play_rect,
                        joint_scores=(live_joint_scores if state.active_wall is not None else None),
                        limb_ema=state.limb_score_ema,
                    )

                state.confetti = update_and_draw_confetti(play_surface, state.confetti, dt)
            elif last_landmarks is not None:
                draw_neon_silhouette(play_surface, last_landmarks, play_rect)

            screen.blit(play_surface, (shake_dx, shake_dy) if state.mode == MODE_PLAYING else (0, 0))

            if camera_frame_for_pip is not None:
                draw_pip(screen, camera_frame_for_pip, screen_h)

            if state.mode == MODE_PLAYING:
                draw_countdown_bar(screen, state.active_wall, state.panel_phase, screen_w)

                score_scale = 1.0
                if state.score_pop_age is not None and state.score_pop_age < SCORE_POP_DURATION:
                    score_scale = lerp(SCORE_POP_SCALE, 1.0, state.score_pop_age / SCORE_POP_DURATION)
                score_surf = render_scaled_text(font_score, f"SCORE: {state.score}", (255, 255, 255), score_scale)
                screen.blit(score_surf, (30, 20))
                if state.floating_score_age is not None and state.floating_score_age < FLOATING_SCORE_DURATION:
                    draw_floating_score(screen, state.floating_score_amount, state.floating_score_age,
                                         (score_surf.get_width() + 46, 20 + score_surf.get_height() // 2), font_floating)
                draw_hearts(screen, state.lives, 3, state.heart_break_index, state.heart_break_age, (30, 74))
                if state.active_wall is not None:
                    label_scale = bounce_in_scale(state.label_bounce_age, LABEL_BOUNCE_DURATION)
                    label_surf = render_scaled_text(font_label, state.active_wall.template.display_label, (255, 255, 0), label_scale)
                    screen.blit(label_surf, label_surf.get_rect(midtop=(screen_w // 2, 20)))

                draw_status_panel(screen, live_score, state.panel_phase, status_panel_center, status_panel_radius, font_status)
                draw_combo_badge(screen, state.combo, state.combo_badge_age,
                                  (status_panel_center[0], status_panel_center[1] - status_panel_radius - 50), font_combo)

                if state.celebration_age is not None and state.celebration_age < CELEBRATION_TEXT_DURATION:
                    draw_scaling_banner(screen, state.celebration_text, state.celebration_age, CELEBRATION_TEXT_DURATION,
                                         (screen_w // 2, screen_h // 2), font_celebration, (255, 225, 60))
                elif state.miss_message_age is not None and state.miss_message_age < MISS_MESSAGE_DURATION:
                    draw_scaling_banner(screen, state.miss_message, state.miss_message_age, MISS_MESSAGE_DURATION,
                                         (screen_w // 2, screen_h // 2), font_celebration, (150, 210, 255))

                if state.flash_timer > 0:
                    alpha = int(FLASH_ALPHA_MAX * (state.flash_timer / FLASH_DURATION))
                    flash_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
                    flash_surf.fill((255, 0, 0, alpha))
                    screen.blit(flash_surf, (0, 0))
                    state.flash_timer -= 1
            elif state.mode == MODE_START:
                draw_start_screen(screen, screen_w, screen_h, state.hold_timer, font_title, font_tagline, font_gate)
            elif state.mode == MODE_GAME_OVER:
                draw_game_over_screen(screen, screen_w, screen_h, state.score, state.walls_cleared,
                                       best_score, state.new_record, state.hold_timer, state.game_over_age,
                                       font_gameover_big, font_gameover_mid, font_gate)

            hint_surf = font_hint.render("Press 'q' or ESC to quit", True, (180, 180, 180))
            screen.blit(hint_surf, (30, screen_h - 40))

            if video_writer is not None:
                frame_rgb = pygame.surfarray.array3d(screen)
                frame_bgr = cv2.cvtColor(frame_rgb.transpose(1, 0, 2), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            pygame.display.flip()

    finally:
        cleanup(cap, video_writer, landmarker)


if __name__ == "__main__":
    main()
