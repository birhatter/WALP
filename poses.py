"""WALP 2 - poses, scoring and pose selection.

Two rules this module exists to enforce:

1. ONE formula. The number the ring shows on screen is the same number that
   decides whether the player passes. There is no hidden second test.
2. Never ask for a pose the camera cannot judge. If the kid's legs are out
   of frame, leg poses are simply not dealt.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import config

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe pose)
# --------------------------------------------------------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

USED_LANDMARKS = (
    NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)

ARM_LANDMARKS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
LEG_LANDMARKS = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

BONES = (
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
)

LIMB_BONES = (
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
)

# Which joint's score tints which bone.
BONE_VERTEX = {
    (L_SHOULDER, L_ELBOW): L_SHOULDER, (L_ELBOW, L_WRIST): L_ELBOW,
    (R_SHOULDER, R_ELBOW): R_SHOULDER, (R_ELBOW, R_WRIST): R_ELBOW,
    (L_HIP, L_KNEE): L_HIP, (L_KNEE, L_ANKLE): L_KNEE,
    (R_HIP, R_KNEE): R_HIP, (R_KNEE, R_ANKLE): R_KNEE,
}

MIRROR_INDEX = {
    NOSE: NOSE,
    L_SHOULDER: R_SHOULDER, R_SHOULDER: L_SHOULDER,
    L_ELBOW: R_ELBOW, R_ELBOW: L_ELBOW,
    L_WRIST: R_WRIST, R_WRIST: L_WRIST,
    L_HIP: R_HIP, R_HIP: L_HIP,
    L_KNEE: R_KNEE, R_KNEE: L_KNEE,
    L_ANKLE: R_ANKLE, R_ANKLE: L_ANKLE,
}

# name -> (a, b, c); the angle is measured at b.
ANGLE_TRIPLES = (
    ("l_elbow", L_SHOULDER, L_ELBOW, L_WRIST),
    ("r_elbow", R_SHOULDER, R_ELBOW, R_WRIST),
    ("l_shoulder", L_HIP, L_SHOULDER, L_ELBOW),
    ("r_shoulder", R_HIP, R_SHOULDER, R_ELBOW),
    ("l_hip", L_SHOULDER, L_HIP, L_KNEE),
    ("r_hip", R_SHOULDER, R_HIP, R_KNEE),
    ("l_knee", L_HIP, L_KNEE, L_ANKLE),
    ("r_knee", R_HIP, R_KNEE, R_ANKLE),
)

ARM_TRIPLES = ("l_elbow", "r_elbow", "l_shoulder", "r_shoulder")
LEG_TRIPLES = ("l_hip", "r_hip", "l_knee", "r_knee")

# Which way each limb POINTS, measured against straight-down.
# Interior joint angles alone cannot tell "arm straight up" from "arm straight
# out to the side" - both are a 180 degree elbow. These specs can, and they
# match how a person actually perceives a pose.
#   0 deg = pointing down, 90 = sideways, 180 = pointing up
AXIS_PAIRS = (
    ("l_upperarm", L_SHOULDER, L_ELBOW),
    ("r_upperarm", R_SHOULDER, R_ELBOW),
    ("l_forearm", L_ELBOW, L_WRIST),
    ("r_forearm", R_ELBOW, R_WRIST),
    ("l_thigh", L_HIP, L_KNEE),
    ("r_thigh", R_HIP, R_KNEE),
    ("l_shin", L_KNEE, L_ANKLE),
    ("r_shin", R_KNEE, R_ANKLE),
    ("torso", L_SHOULDER, L_HIP),
    ("torso_r", R_SHOULDER, R_HIP),
)

ARM_SPECS = ARM_TRIPLES + ("l_upperarm", "r_upperarm", "l_forearm", "r_forearm")
# Which limb each measurement belongs to. Grouped per limb rather than
# per half: in a tree pose one leg stays straight, so "legs" as a single
# group is half-correct for free and the raised leg can be skipped.
SPEC_GROUP = {}
for _n in ("l_elbow", "l_shoulder", "l_upperarm", "l_forearm"):
    SPEC_GROUP[_n] = "l_arm"
for _n in ("r_elbow", "r_shoulder", "r_upperarm", "r_forearm"):
    SPEC_GROUP[_n] = "r_arm"
LEG_SPECS = LEG_TRIPLES + ("l_thigh", "r_thigh", "l_shin", "r_shin")
for _n in ("l_hip", "l_knee", "l_thigh", "l_shin"):
    SPEC_GROUP[_n] = "l_leg"
for _n in ("r_hip", "r_knee", "r_thigh", "r_shin"):
    SPEC_GROUP[_n] = "r_leg"
SPEC_GROUP["torso"] = SPEC_GROUP["torso_r"] = "torso"
THIGH_SPECS = ("l_hip", "r_hip", "l_thigh", "r_thigh")
TORSO_SPECS = ("torso", "torso_r")


def angle_at(a, b, c) -> float:
    """Unsigned angle in degrees at point b."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(abs(cross), dot))


def axis_angle(a, b) -> float:
    """Unsigned angle of the vector a->b away from straight down."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(abs(dx), dy if dy != 0 else 1e-6))


# --------------------------------------------------------------------------
# Pose library
#
# Joints are given in BODY UNITS: x and y share one scale, y=0 near the top
# of the head and y~0.95 at the feet. The builder works out the bounding box
# and the aspect ratio of the hole automatically, so a wide pose gets a wide
# wall without any hand-tuned numbers.
# --------------------------------------------------------------------------

def _body(over: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[int, Tuple[float, float]]:
    """Neutral standing figure with the given joints moved.

    Takes a dict keyed by the landmark constants, not keyword arguments -
    keyword names would arrive as strings and silently fail to override.
    """
    base = {
        NOSE: (0.500, 0.130),
        L_SHOULDER: (0.620, 0.290), R_SHOULDER: (0.380, 0.290),
        L_ELBOW: (0.640, 0.420), R_ELBOW: (0.360, 0.420),
        L_WRIST: (0.655, 0.545), R_WRIST: (0.345, 0.545),
        L_HIP: (0.575, 0.550), R_HIP: (0.425, 0.550),
        L_KNEE: (0.565, 0.750), R_KNEE: (0.435, 0.750),
        L_ANKLE: (0.555, 0.940), R_ANKLE: (0.445, 0.940),
    }
    if over:
        unknown = set(over) - set(base)
        if unknown:
            raise KeyError(f"unknown landmark keys in pose override: {unknown}")
        base.update(over)
    return base


POSE_LIBRARY = [
    # ---- tier 1: arms only, feet may be off camera ----
    dict(name="T_POSE", label="T-POSE", tier=1, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.790, 0.290), R_ELBOW: (0.210, 0.290),
                       L_WRIST: (0.960, 0.290), R_WRIST: (0.040, 0.290)})),
    dict(name="CHEER", label="CHEER!", tier=1,
         required=("l_shoulder", "r_shoulder", "l_upperarm", "r_upperarm",
                   "l_forearm", "r_forearm"),
         joints=_body({L_ELBOW: (0.645, 0.155), R_ELBOW: (0.355, 0.155),
                       L_WRIST: (0.660, 0.030), R_WRIST: (0.340, 0.030)})),
    dict(name="Y_POSE", label="Y-POSE", tier=1, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.760, 0.195), R_ELBOW: (0.240, 0.195),
                       L_WRIST: (0.900, 0.100), R_WRIST: (0.100, 0.100)})),
    dict(name="HANDS_ON_HEAD", label="HANDS ON HEAD", tier=1, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.800, 0.225), R_ELBOW: (0.200, 0.225),
                       L_WRIST: (0.565, 0.105), R_WRIST: (0.435, 0.105)})),
    dict(name="ONE_ARM_UP", label="ONE ARM UP", tier=1, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.680, 0.175), R_ELBOW: (0.355, 0.425),
                       L_WRIST: (0.720, 0.045), R_WRIST: (0.340, 0.560)})),

    # ---- tier 2 ----
    dict(name="L_POSE", label="L-POSE", tier=2, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.628, 0.165), R_ELBOW: (0.210, 0.290),
                       L_WRIST: (0.635, 0.040), R_WRIST: (0.040, 0.290)})),
    dict(name="FLEX", label="STRONG ARMS", tier=2, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.805, 0.300), R_ELBOW: (0.195, 0.300),
                       L_WRIST: (0.700, 0.130), R_WRIST: (0.300, 0.130)})),
    dict(name="WAVE", label="BIG WAVE", tier=2, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.745, 0.300), R_ELBOW: (0.355, 0.425),
                       L_WRIST: (0.745, 0.120), R_WRIST: (0.340, 0.560)})),
    dict(name="ROBOT", label="ROBOT", tier=2, required=ARM_SPECS,
         joints=_body({L_ELBOW: (0.780, 0.290), R_ELBOW: (0.220, 0.290),
                       L_WRIST: (0.780, 0.125), R_WRIST: (0.220, 0.455)})),

    # ---- tier 3: legs needed ----
    dict(name="X_POSE", label="X-POSE", tier=3, required=ARM_SPECS + LEG_SPECS,
         joints=_body({L_ELBOW: (0.775, 0.175), R_ELBOW: (0.225, 0.175),
                       L_WRIST: (0.910, 0.070), R_WRIST: (0.090, 0.070),
                       L_KNEE: (0.730, 0.730), R_KNEE: (0.270, 0.730),
                       L_ANKLE: (0.880, 0.900), R_ANKLE: (0.120, 0.900)})),
    dict(name="SQUAT", label="SQUAT", tier=3, required=LEG_SPECS + TORSO_SPECS,
         tol_mult=1.25,
         joints={NOSE: (0.500, 0.280),
                 L_SHOULDER: (0.620, 0.440), R_SHOULDER: (0.380, 0.440),
                 L_ELBOW: (0.745, 0.540), R_ELBOW: (0.255, 0.540),
                 L_WRIST: (0.800, 0.625), R_WRIST: (0.200, 0.625),
                 L_HIP: (0.625, 0.635), R_HIP: (0.375, 0.635),
                 L_KNEE: (0.700, 0.790), R_KNEE: (0.300, 0.790),
                 L_ANKLE: (0.600, 0.940), R_ANKLE: (0.400, 0.940)}),
    dict(name="SIDE_KICK", label="SIDE KICK", tier=3,
         required=LEG_SPECS + ("l_shoulder", "r_shoulder", "l_upperarm", "r_upperarm"),
         tol_mult=1.15,
         joints=_body({L_ELBOW: (0.760, 0.385), R_ELBOW: (0.240, 0.385),
                       L_WRIST: (0.865, 0.300), R_WRIST: (0.135, 0.300),
                       L_KNEE: (0.700, 0.625), R_KNEE: (0.435, 0.750),
                       L_ANKLE: (0.845, 0.690), R_ANKLE: (0.445, 0.940)})),
    dict(name="SUPERHERO", label="SUPERHERO", tier=3, required=ARM_SPECS + THIGH_SPECS,
         joints=_body({L_ELBOW: (0.675, 0.180), R_ELBOW: (0.245, 0.400),
                       L_WRIST: (0.715, 0.050), R_WRIST: (0.400, 0.530),
                       L_KNEE: (0.700, 0.740), R_KNEE: (0.300, 0.740),
                       L_ANKLE: (0.820, 0.930), R_ANKLE: (0.180, 0.930)})),

    # ---- tier 4: the hard ones ----
    dict(name="TREE", label="TREE POSE", tier=4,
         required=ARM_SPECS + LEG_SPECS, tol_mult=1.2,
         joints=_body({L_SHOULDER: (0.600, 0.285), R_SHOULDER: (0.400, 0.285),
                       L_ELBOW: (0.580, 0.155), R_ELBOW: (0.420, 0.155),
                       L_WRIST: (0.515, 0.035), R_WRIST: (0.485, 0.035),
                       L_KNEE: (0.690, 0.700), R_KNEE: (0.435, 0.750),
                       L_ANKLE: (0.505, 0.665), R_ANKLE: (0.445, 0.940)})),
    dict(name="RUNNER", label="RUNNER", tier=4,
         required=ARM_SPECS + LEG_SPECS, tol_mult=1.2,
         joints=_body({L_ELBOW: (0.700, 0.225), R_ELBOW: (0.305, 0.440),
                       L_WRIST: (0.745, 0.100), R_WRIST: (0.265, 0.565),
                       L_KNEE: (0.565, 0.750), R_KNEE: (0.400, 0.575),
                       L_ANKLE: (0.555, 0.940), R_ANKLE: (0.520, 0.700)})),
    dict(name="LEAN", label="SIDE STRETCH", tier=4,
         required=TORSO_SPECS + ("l_shoulder", "r_shoulder", "l_upperarm",
                                 "r_upperarm", "l_forearm", "r_forearm"),
         tol_mult=1.15,
         joints={NOSE: (0.330, 0.175),
                 L_SHOULDER: (0.430, 0.330), R_SHOULDER: (0.240, 0.370),
                 L_ELBOW: (0.360, 0.185), R_ELBOW: (0.185, 0.500),
                 L_WRIST: (0.255, 0.090), R_WRIST: (0.175, 0.620),
                 L_HIP: (0.625, 0.560), R_HIP: (0.490, 0.585),
                 L_KNEE: (0.610, 0.755), R_KNEE: (0.500, 0.760),
                 L_ANKLE: (0.600, 0.940), R_ANKLE: (0.505, 0.940)}),
]

# Breathing room around the hole, in body units. Must clear half a limb
# (LIMB_BODY_UNITS / 2) and the head radius, or hands and heads get
# sliced off by the edge of the wall.
BOX_PAD = 0.105
# Never cut a wall narrower than this, or a narrow pose leaves a sliver
# of wall with nothing to smash.
MIN_BOX_ASPECT = 0.62


@dataclass(frozen=True)
class AngleSpec:
    name: str
    a: int
    b: int
    c: int
    target: float
    kind: str = "joint"   # "joint" or "axis" (axis ignores c)


@dataclass(frozen=True)
class PoseTemplate:
    name: str
    label: str
    tier: int
    joints: Dict[int, Tuple[float, float]]      # normalised into the hole box
    specs: Tuple[AngleSpec, ...]
    required: Tuple[str, ...]
    box_aspect: float
    unit_h: float          # box height in body units, for limb sizing
    tol_mult: float
    cue_index: int
    is_custom: bool = False

    def required_landmarks(self) -> Tuple[int, ...]:
        needed = {L_SHOULDER, R_SHOULDER, L_HIP, R_HIP}
        for spec in self.specs:
            if spec.name not in self.required:
                continue
            needed.add(spec.a)
            needed.add(spec.b)
            if spec.kind == "joint":
                needed.add(spec.c)
        return tuple(sorted(needed))

    def needs_legs(self) -> bool:
        return any(i in LEG_LANDMARKS and i not in (L_HIP, R_HIP)
                   for i in self.required_landmarks())


def _normalise(joints: Dict[int, Tuple[float, float]]):
    xs = [p[0] for p in joints.values()]
    ys = [p[1] for p in joints.values()]
    min_x, max_x = min(xs) - BOX_PAD, max(xs) + BOX_PAD
    min_y, max_y = min(ys) - BOX_PAD, max(ys) + BOX_PAD
    w = max(max_x - min_x, 1e-3)
    h = max(max_y - min_y, 1e-3)
    if w / h < MIN_BOX_ASPECT:
        # widen the box symmetrically; the figure keeps its proportions
        extra = (h * MIN_BOX_ASPECT - w) / 2.0
        min_x -= extra
        max_x += extra
        w = max_x - min_x
    norm = {i: ((x - min_x) / w, (y - min_y) / h) for i, (x, y) in joints.items()}
    return norm, w / h, h


def build_template(spec_dict: dict, cue_index: int, is_custom: bool = False) -> PoseTemplate:
    raw = spec_dict["joints"]
    norm, aspect, unit_h = _normalise(raw)

    specs: List[AngleSpec] = []
    for name, a, b, c in ANGLE_TRIPLES:
        specs.append(AngleSpec(name, a, b, c, angle_at(raw[a], raw[b], raw[c]), "joint"))
    for name, a, b in AXIS_PAIRS:
        specs.append(AngleSpec(name, a, b, b, axis_angle(raw[a], raw[b]), "axis"))

    required = tuple(spec_dict.get("required") or ARM_SPECS)
    known = {s.name for s in specs}
    unknown = set(required) - known
    if unknown:
        raise KeyError(f"{spec_dict['name']}: unknown spec names {unknown}")

    return PoseTemplate(
        name=spec_dict["name"],
        label=spec_dict["label"],
        tier=int(spec_dict.get("tier", 1)),
        joints=norm,
        specs=tuple(specs),
        required=required,
        box_aspect=aspect,
        unit_h=unit_h,
        tol_mult=float(spec_dict.get("tol_mult", 1.0)),
        cue_index=cue_index,
        is_custom=is_custom,
    )


PRESET_TEMPLATES: List[PoseTemplate] = [
    build_template(d, i) for i, d in enumerate(POSE_LIBRARY)
]
BY_NAME = {t.name: t for t in PRESET_TEMPLATES}
T_POSE = BY_NAME["T_POSE"]


def mirrored(specs: Sequence[AngleSpec]) -> Tuple[AngleSpec, ...]:
    return tuple(
        AngleSpec(s.name, MIRROR_INDEX[s.a], MIRROR_INDEX[s.b],
                  MIRROR_INDEX[s.c], s.target, s.kind)
        for s in specs
    )


# --------------------------------------------------------------------------
# Scoring - the single source of truth
# --------------------------------------------------------------------------

def _point(landmarks, idx: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return (lm.x * 1000.0, lm.y * 1000.0)


def _visible(landmarks, idx: int) -> bool:
    vis = getattr(landmarks[idx], "visibility", None)
    return vis is None or vis >= config.VISIBILITY_THRESHOLD


def _measure(landmarks, spec: AngleSpec) -> Optional[float]:
    needed = (spec.a, spec.b) if spec.kind == "axis" else (spec.a, spec.b, spec.c)
    for idx in needed:
        if not _visible(landmarks, idx):
            return None
    if spec.kind == "axis":
        return axis_angle(_point(landmarks, spec.a), _point(landmarks, spec.b))
    return angle_at(_point(landmarks, spec.a), _point(landmarks, spec.b),
                    _point(landmarks, spec.c))


def _score_side(specs, required, landmarks, tolerance):
    """Average score over the required specs, plus a per-joint breakdown."""
    zero_at = tolerance * config.SCORE_ZERO_ERROR_MULT
    by_name: Dict[str, float] = {}
    for spec in specs:
        angle = _measure(landmarks, spec)
        if angle is None:
            by_name[spec.name] = 0.0
        else:
            err = abs(angle - spec.target)
            by_name[spec.name] = max(0.0, 1.0 - err / max(zero_at, 1e-6))

    req = [n for n in required if n in by_name]
    if req:
        mean = sum(by_name[n] for n in req) / len(req)
        # Weakest-link term. A plain average let you nail the arms of a tree
        # pose and ignore the raised leg entirely: the perfect arm
        # measurements simply outvoted the wrong leg ones. Mixing in the
        # worst-matching limb means every limb the pose asks for has to be
        # attempted, while still tolerating one limb being tracked poorly.
        groups: Dict[str, List[float]] = {}
        for n in req:
            groups.setdefault(SPEC_GROUP.get(n, "other"), []).append(by_name[n])
        worst = min(sum(v) / len(v) for v in groups.values())
        total = mean * (1.0 - config.WEAKEST_LINK_WEIGHT) + worst * config.WEAKEST_LINK_WEIGHT
    else:
        total = 0.0

    joint_scores: Dict[int, float] = {}
    for spec in specs:
        if spec.kind == "joint":
            joint_scores[spec.b] = by_name[spec.name]
    return total, joint_scores


def graded_score(template: PoseTemplate, landmarks,
                 tolerance_scale: float = 1.0) -> Tuple[float, Dict[int, float]]:
    """Returns (0..1 score, per-joint scores). Mirror-friendly: a kid raising
    the 'wrong' arm still counts, which matters a lot at this age."""
    if landmarks is None:
        return 0.0, {}
    tol = config.BASE_TOLERANCE_DEG * template.tol_mult * tolerance_scale
    straight, js_straight = _score_side(template.specs, template.required, landmarks, tol)
    flipped, js_flipped = _score_side(mirrored(template.specs), template.required, landmarks, tol)
    if straight >= flipped:
        return straight, js_straight
    return flipped, js_flipped


# --------------------------------------------------------------------------
# Visibility tracking -> which poses are fair to ask for right now
# --------------------------------------------------------------------------

class VisibilityTracker:
    """Rolling confidence that each landmark is actually in frame."""

    def __init__(self) -> None:
        self.avg: Dict[int, float] = {i: 0.0 for i in USED_LANDMARKS}
        self._warm = 0

    def update(self, landmarks) -> None:
        alpha = 2.0 / (config.VISIBILITY_HISTORY + 1)
        for idx in USED_LANDMARKS:
            if landmarks is None:
                v = 0.0
            else:
                raw = getattr(landmarks[idx], "visibility", None)
                v = 1.0 if raw is None else float(raw)
            self.avg[idx] = self.avg[idx] + (v - self.avg[idx]) * alpha
        self._warm = min(self._warm + 1, config.VISIBILITY_HISTORY)

    @property
    def warmed_up(self) -> bool:
        return self._warm >= config.VISIBILITY_HISTORY // 3

    def can_see(self, idx: int) -> bool:
        return self.avg.get(idx, 0.0) >= config.SPAWNABLE_VISIBILITY

    def sees_legs(self) -> bool:
        return all(self.can_see(i) for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE))

    def sees_arms(self) -> bool:
        return all(self.can_see(i) for i in ARM_LANDMARKS)

    def supports(self, template: PoseTemplate) -> bool:
        return all(self.can_see(i) for i in template.required_landmarks())


class PoseDealer:
    """Deals the next pose: right difficulty, actually visible, not a repeat."""

    def __init__(self, custom: Optional[List[PoseTemplate]] = None) -> None:
        self.custom = custom or []
        self.recent: List[str] = []
        self.dealt = 0
        self.last_seen: Dict[str, int] = {}

    def all_templates(self) -> List[PoseTemplate]:
        return PRESET_TEMPLATES + self.custom

    def available(self, walls_cleared: int, vis: VisibilityTracker) -> List[PoseTemplate]:
        tier_cap = config.max_tier(walls_cleared)
        pool = [t for t in self.all_templates()
                if t.tier <= tier_cap and vis.supports(t)]
        if not pool:
            pool = [t for t in self.all_templates() if vis.supports(t)]
        if not pool:
            pool = [T_POSE]
        return pool

    def deal(self, walls_cleared: int, vis: VisibilityTracker) -> PoseTemplate:
        """Pick the next pose, weighted by how long since it last came up.

        Plain random clumps: the same shape three times in five feels broken.
        Cycling through a shuffled list fixes the clumping but becomes a
        pattern you can learn. Weighting by staleness avoids both - recent
        shapes are unlikely, long-unseen ones are likely, and nothing is ever
        guaranteed.
        """
        pool = self.available(walls_cleared, vis)
        self.dealt += 1

        weights = []
        for t in pool:
            since = self.dealt - self.last_seen.get(t.name, -999)
            # never the same shape twice in a row, and rarely within three
            weights.append(0.0 if since <= 1 else float(since - 1) ** 1.6)
        if sum(weights) <= 0.0:
            weights = [1.0] * len(pool)

        chosen = random.choices(pool, weights=weights, k=1)[0]
        self.last_seen[chosen.name] = self.dealt
        self.recent.append(chosen.name)
        del self.recent[:-6]
        return chosen


# --------------------------------------------------------------------------
# Custom poses ("MY POSES" studio)
# --------------------------------------------------------------------------

def capture_custom(landmarks, label: str) -> Optional[dict]:
    """Turn a live skeleton into a reusable pose definition."""
    if landmarks is None:
        return None
    pts = {}
    for idx in USED_LANDMARKS:
        if not _visible(landmarks, idx):
            return None
        # The play field is mirrored, so mirror x here too. That way the
        # saved hole matches what the child saw themselves doing.
        pts[idx] = (1.0 - landmarks[idx].x, landmarks[idx].y)

    ys = [p[1] for p in pts.values()]
    height = max(max(ys) - min(ys), 1e-3)
    cx = (pts[L_HIP][0] + pts[R_HIP][0]) / 2.0
    top = min(ys)
    # Rescale into body units so it sits alongside the built-in poses.
    scaled = {
        i: (0.5 + (x - cx) / height * 0.81,
            0.13 + (y - top) / height * 0.81)
        for i, (x, y) in pts.items()
    }
    return {
        "name": f"CUSTOM_{label}",
        "label": label.upper(),
        "tier": 2,
        "required": list(ARM_SPECS),
        "tol_mult": 1.25,
        "joints": {str(i): list(p) for i, p in scaled.items()},
    }


def _decode(entry: dict) -> dict:
    out = dict(entry)
    out["joints"] = {int(k): tuple(v) for k, v in entry["joints"].items()}
    out["required"] = tuple(entry.get("required") or ARM_SPECS)
    return out


def load_custom() -> List[PoseTemplate]:
    if not os.path.exists(config.CUSTOM_POSES_PATH):
        return []
    try:
        with open(config.CUSTOM_POSES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for i, entry in enumerate(data if isinstance(data, list) else []):
        try:
            out.append(build_template(_decode(entry), len(POSE_LIBRARY) + i, is_custom=True))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_custom(entries: List[dict]) -> bool:
    try:
        with open(config.CUSTOM_POSES_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
        return True
    except OSError:
        return False


def load_custom_raw() -> List[dict]:
    if not os.path.exists(config.CUSTOM_POSES_PATH):
        return []
    try:
        with open(config.CUSTOM_POSES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []
