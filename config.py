"""WALP 2 - central tuning file.

Everything a grown-up might want to tweak lives here.
Nothing in this file does any work; it is pure numbers and colours.
"""

import math
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
SAVE_PATH = os.path.join(ROOT_DIR, "walp2_save.json")
CUSTOM_POSES_PATH = os.path.join(ROOT_DIR, "my_poses.json")

MODEL_CANDIDATES = (
    os.path.join(ASSETS_DIR, "pose_landmarker_lite.task"),
    os.path.join(ASSETS_DIR, "pose_landmarker_full.task"),
)

# --------------------------------------------------------------------------
# Screen recording
#
# Records the whole game screen, which includes the camera view - so these
# files contain video of whoever is playing. They stay in RECORD_DIR and are
# never uploaded anywhere. Set RECORD_AUTO_RUNS to False to only record when
# you press R.
# --------------------------------------------------------------------------
# Master switch. False turns recording off completely: no folder is made, no
# clip is started with a run, and the R key does nothing. Everything else
# below is left in place and works again the moment this goes back to True.
RECORDING_ENABLED = False
RECORD_DIR = os.path.join(ROOT_DIR, "recordings")
RECORD_AUTO_RUNS = True      # one clip per attempt, started and stopped for you
RECORD_FPS = 30              # recorded fps, independent of the game's 60
RECORD_MAX_WIDTH = 1280      # downscale before encoding, keeps files sane
RECORD_QUEUE = 6             # frames buffered; over this, frames are dropped
# Make the folder as soon as the game starts, so it is visibly there
# rather than appearing only after the first run finishes.
RECORD_CREATE_DIR_ON_START = True

# On-screen name of the game.
GAME_TITLE = "WALP"

# --------------------------------------------------------------------------
# Display / timing
# --------------------------------------------------------------------------
TARGET_FPS = 60
CAMERA_W, CAMERA_H = 1280, 720
CAMERA_FPS = 30

# Reference resolution the layout was designed against. All sizes scale
# from this so the game looks right on a small laptop or a big TV.
DESIGN_W, DESIGN_H = 1600, 900

# --------------------------------------------------------------------------
# Pose detection
# --------------------------------------------------------------------------
VISIBILITY_THRESHOLD = 0.35
# A joint must be this visible for a pose that *needs* it to be spawnable.
SPAWNABLE_VISIBILITY = 0.55
# How many recent frames we look at when deciding if a body part is reliable.
VISIBILITY_HISTORY = 45

# The filled body silhouette. It looks nice but costs inference time on
# the tracking thread AND a full-window rescale when drawing, and the
# tracker is what decides how quickly the game reacts to you. The neon
# outline already shows where your body is, so this is off by default.
# Turn it on if your machine has CPU to spare.
SEGMENTATION_ENABLED = False
SEGMENTATION_DOWNSCALE = 4  # mask is rendered at 1/N res then smooth-scaled up

# --------------------------------------------------------------------------
# Scoring rules
#
# ONE formula decides everything the player sees AND whether they pass.
# The ring on screen is literally the rule.
# --------------------------------------------------------------------------
PASS_THRESHOLD = 0.70      # ring this full  -> you go through, full points
CRACK_THRESHOLD = 0.45     # ring this full  -> you squeeze through, half points
BASE_TOLERANCE_DEG = 35.0
SCORE_ZERO_ERROR_MULT = 1.5
# How much the worst-matching body part counts, versus the average of
# everything. At 0 you can pass a pose by doing only the arms; higher
# values force you to attempt every part of the shape.
WEAKEST_LINK_WEIGHT = 0.35

# Smoothing on the live score. Higher = snappier, lower = steadier.
# Kept high on purpose: this game is about reacting quickly, and a
# score that lags your body feels like the game is ignoring you.
LIVE_SCORE_EMA_ALPHA = 0.60
LIMB_SCORE_EMA_ALPHA = 0.50

SCORE_PER_HIT = 100
SCORE_PER_CRACK = 50
MAX_COMBO_MULTIPLIER = 5

# --------------------------------------------------------------------------
# Difficulty curve
#
# Design goal: a 5 year old clears the first several walls easily, and
# *nobody* survives forever. Every knob keeps tightening with no plateau
# that a player can camp on.
# --------------------------------------------------------------------------
TRAVEL_START = 5.2         # seconds for wall 1 to reach the player
TRAVEL_MIN = 0.85          # asymptote - approached but never quite reached
TRAVEL_DECAY = 15.0

JUDGE_START_EASY = 0.55    # judging opens early = wide forgiving window
JUDGE_START_HARD = 0.93
JUDGE_DECAY = 20.0

TOLERANCE_SCALE_EASY = 1.45   # 35 deg * 1.45 = ~50 deg of slack. Very kind.
TOLERANCE_SCALE_HARD = 0.55   # 35 deg * 0.55 = ~19 deg. Brutal.
TOLERANCE_DECAY = 17.0

# The exponentials above flatten out, which would let a player who never
# misses go on forever. Past this wall count everything keeps tightening
# linearly instead, and the "squeeze through" rescue is withdrawn, so every
# run has an end.
LATE_WALL = 55
LATE_TRAVEL_DROP = 0.010      # seconds removed per wall past LATE_WALL
TRAVEL_HARD_FLOOR = 0.35
CRACK_FADE_WALLS = 45.0       # partial credit is gone by this wall

# Poses unlock in tiers so wall #1 is never a one-legged kick.
TIER_UNLOCK_WALL = {1: 0, 2: 4, 3: 10, 4: 18}

# The wall never moves sideways and never chases you. One player gets it dead
# centre; two players split the screen exactly in half, one wall per side.
# Where you stand in the room changes nothing about the run - the game is
# about changing shape on the spot, not about running around.
WALL_MOVES_SIDEWAYS = False
ONE_PLAYER_WALL_X = 0.50
TWO_PLAYER_WALL_X = (0.25, 0.75)

OFFCENTER_START_WALL = 8      # only used when WALL_MOVES_SIDEWAYS is on
OFFCENTER_MAX = 0.30          # +/- fraction of screen width
OFFCENTER_RAMP_WALLS = 25.0

START_LIVES = 3
MAX_LIVES = 3                 # three hearts, no more
BONUS_HEART_EVERY = 10        # clear this many walls -> get one back
# No cap on how often a life comes back: by the time it would matter the
# walls are arriving faster than one life per ten could ever cover.
MAX_BONUS_HEARTS = 99

WALLS_PER_WORLD = 6           # background theme changes this often


def travel_seconds(walls_cleared: int) -> float:
    """Seconds for a wall to travel from the horizon to the player."""
    span = TRAVEL_START - TRAVEL_MIN
    base = TRAVEL_MIN + span * math.exp(-walls_cleared / TRAVEL_DECAY)
    if walls_cleared > LATE_WALL:
        base -= (walls_cleared - LATE_WALL) * LATE_TRAVEL_DROP
    return max(TRAVEL_HARD_FLOOR, base)


def crack_threshold(walls_cleared: int) -> float:
    """How good a near-miss has to be to still squeeze through.

    Starts generous so small children keep their hearts while learning, then
    climbs to meet the pass mark - by which point there is no safety net.
    """
    t = min(1.0, walls_cleared / CRACK_FADE_WALLS)
    return CRACK_THRESHOLD + (PASS_THRESHOLD - CRACK_THRESHOLD) * t


def judgment_start(walls_cleared: int) -> float:
    """Progress fraction at which a pose starts counting."""
    span = JUDGE_START_HARD - JUDGE_START_EASY
    return JUDGE_START_HARD - span * math.exp(-walls_cleared / JUDGE_DECAY)


def tolerance_scale(walls_cleared: int) -> float:
    """Multiplier applied to every joint's angle tolerance."""
    span = TOLERANCE_SCALE_EASY - TOLERANCE_SCALE_HARD
    return TOLERANCE_SCALE_HARD + span * math.exp(-walls_cleared / TOLERANCE_DECAY)


def offcenter_amount(walls_cleared: int) -> float:
    """How far sideways a wall may sit, as a fraction of screen width."""
    if not WALL_MOVES_SIDEWAYS:
        return 0.0
    if walls_cleared < OFFCENTER_START_WALL:
        return 0.0
    t = min(1.0, (walls_cleared - OFFCENTER_START_WALL) / OFFCENTER_RAMP_WALLS)
    return OFFCENTER_MAX * t


def max_tier(walls_cleared: int) -> int:
    tier = 1
    for t, wall in TIER_UNLOCK_WALL.items():
        if walls_cleared >= wall:
            tier = max(tier, t)
    return tier


def world_index(walls_cleared: int) -> int:
    return walls_cleared // WALLS_PER_WORLD


# --------------------------------------------------------------------------
# Wall geometry & feel
# --------------------------------------------------------------------------
# Height the wall appears at. Small values make the shape unreadable
# until it is nearly on top of you, which leaves no time to react.
WALL_FAR_H_FRAC = 0.40        # height at the horizon, fraction of screen
WALL_FALLBACK_H_FRAC = 0.82   # used when we cannot see the player yet
# How much bigger than you the wall gets. Above 1.0 the hole is roomier
# than your body, which is what makes fitting through feel fair.
WALL_PLAYER_MARGIN = 1.30
WALL_MAX_H_FRAC = 0.92        # never taller than the screen
JUDGMENT_LINE_Y_FRAC = 0.70
WALL_TOP_Y_FRAC = 0.21        # keeps the wall fully on screen at spawn
WALL_PASS_PROGRESS = 1.45     # walls keep going past the line and off-screen
WALL_PASS_SECONDS = 0.26      # how long the fly-past takes

WALL_CORNER_RADIUS_FRAC = 0.09
# Limb and head size are given in BODY units, not as a fraction of the
# wall. A squat is a short wall around a full-size person, so sizing off
# the wall made its limbs enormous and the hole melted into a blob.
LIMB_BODY_UNITS = 0.150       # limb diameter
HEAD_BODY_UNITS = 0.088       # head radius
WALL_LIMB_THICKNESS_FRAC = 0.205   # legacy fallback
WALL_HEAD_RADIUS_FRAC = 0.105
WALL_OUTLINE_PX = 7

SPAWN_GAP_SECONDS = 0.32      # dead air between walls - keep this tight
HIT_CELEBRATE_SECONDS = 0.5
MISS_LINGER_SECONDS = 0.45

SHARD_COUNT = 26
SHARD_GRAVITY = 1500.0
SHARD_LIFETIME = 1.1
SHARD_SPIN_MAX = 9.0

# --------------------------------------------------------------------------
# Camera view (picture-in-picture)
#
# Showing the player their own camera feed is display only - it is never
# written anywhere.
# --------------------------------------------------------------------------
PIP_ENABLED = True
PIP_SRC_W, PIP_SRC_H = 320, 180      # size the worker downscales to
PIP_HEIGHT_FRAC = 0.155              # on-screen height, fraction of the window
PIP_MARGIN_FRAC = 0.025
PIP_CALIBRATE_HEIGHT_FRAC = 0.34     # bigger while you are lining yourself up

# --------------------------------------------------------------------------
# Touchless hand-cursor UI
# --------------------------------------------------------------------------
HOVER_SECONDS = 0.75          # hold a hand over a button this long to press
CURSOR_SMOOTHING = 0.55
CURSOR_RADIUS_FRAC = 0.028
BUTTON_RADIUS_FRAC = 0.098
# Where the button row sits. Kept clear of the bottom of the screen,
# which is where an arm hanging at rest points.
BUTTON_ROW_Y = 0.55

POSE_HOLD_SECONDS = 0.7       # T-pose quick-start
# Keep the pointer alive briefly after the hand is lost, so a
# momentary tracking dropout does not cancel a selection.
CURSOR_GRACE_SECONDS = 0.5
# Selection snaps to the nearest button, so pointing roughly at one
# is enough - this is what makes it usable from across a room.
BUTTON_SNAP = True

# Menu control by raising an arm. A yes/no "is this wrist above this
# shoulder" is enormously more reliable than tracking a hand position, which
# is why this is the primary way to drive the menus.
GESTURE_MENU = False   # arm-step menu; the free hand pointer is primary
GESTURE_FIRST_STEP = 0.22     # delay before the first move, stops double steps
GESTURE_REPEAT = 0.50         # hold an arm up to keep stepping
GESTURE_CONFIRM_SECONDS = 0.6  # both arms up this long = select

# --------------------------------------------------------------------------
# Calibration (make sure the camera can actually see the whole kid)
# --------------------------------------------------------------------------
CALIB_REQUIRED_SECONDS = 1.0
# Never trap anyone on the framing screen.
CALIB_AUTO_SKIP_SECONDS = 12.0
CALIB_MARGIN = 0.045          # body must sit this far inside the frame edges

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------
INK = (18, 16, 30)            # near-black used for every outline
CREAM = (238, 242, 255)
SUNNY = (255, 196, 44)
CORAL = (255, 74, 105)
MINT = (54, 226, 156)
SKY = (58, 170, 255)
GRAPE = (150, 92, 255)

SCORE_GOOD = (54, 226, 156)
SCORE_OK = (255, 196, 44)
SCORE_BAD = (255, 74, 105)

HEART_FULL = (255, 74, 105)
HEART_EMPTY = (74, 70, 102)

PLAYER_SKINS = [
    ("CYAN",    (64, 226, 255), 0),
    ("MAGENTA", (255, 78, 205), 400),
    ("LIME",    (150, 255, 74), 1500),
    ("AMBER",   (255, 170, 40), 3500),
    ("VIOLET",  (168, 110, 255), 7000),
    ("SPECTRUM", None, 14000),        # None = animated hue cycle
]

# --------------------------------------------------------------------------
# World themes
#
# Deep, saturated backdrops with neon accents - the wall and the player glow
# have to pop off them, and washed-out pastels read as a game for infants.
# --------------------------------------------------------------------------
WORLDS = [
    {
        "name": "NEON CITY",
        "sky_top": (26, 20, 58), "sky_bottom": (96, 40, 122),
        "hills": [(58, 38, 96), (42, 27, 74), (28, 18, 52)],
        "motif": "star", "accent": (64, 226, 255), "root": 0,
    },
    {
        "name": "CANYON RUN",
        "sky_top": (52, 22, 44), "sky_bottom": (186, 78, 62),
        "hills": [(122, 54, 62), (92, 40, 52), (62, 27, 40)],
        "motif": "leaf", "accent": (255, 196, 44), "root": 2,
    },
    {
        "name": "DEEP SPACE",
        "sky_top": (8, 8, 26), "sky_bottom": (58, 30, 104),
        "hills": [(42, 30, 80), (28, 20, 58), (16, 12, 38)],
        "motif": "star", "accent": (255, 78, 205), "root": 5,
    },
    {
        "name": "ABYSS",
        "sky_top": (4, 32, 60), "sky_bottom": (16, 104, 132),
        "hills": [(18, 76, 104), (12, 56, 80), (8, 38, 58)],
        "motif": "bubble", "accent": (54, 226, 156), "root": 7,
    },
    {
        "name": "MAGMA",
        "sky_top": (40, 12, 22), "sky_bottom": (156, 44, 30),
        "hills": [(104, 34, 32), (74, 24, 26), (46, 15, 20)],
        "motif": "candy", "accent": (255, 150, 40), "root": 9,
    },
    {
        "name": "GLACIER",
        "sky_top": (18, 40, 72), "sky_bottom": (76, 152, 190),
        "hills": [(70, 116, 156), (48, 86, 122), (30, 58, 90)],
        "motif": "snow", "accent": (120, 232, 255), "root": 4,
    },
]

# --------------------------------------------------------------------------
# Encouragement copy. Deliberately never says "wrong" or "fail".
# --------------------------------------------------------------------------
HIT_WORDS = ("PERFECT", "CLEAN", "NAILED IT", "FLAWLESS", "SHARP", "LOCKED IN")
CRACK_WORDS = ("SCRAPED THROUGH", "CLOSE ONE", "JUST MADE IT", "TIGHT")
MISS_WORDS = ("MISSED", "TOO SLOW", "NOT QUITE", "OFF SHAPE")
GAMEOVER_WORDS = ("RUN OVER", "GOOD RUN", "NICE RUN", "THAT'S THE RUN")

STAR_THRESHOLDS = (4, 10, 18)

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 44100
# Mixer buffer, in samples. Small is what keeps the hit and miss sounds tight
# against the wall. Linux gets a bigger one: SDL's PulseAudio/PipeWire backend
# halves whatever is asked for, and at 512 its sound thread starves while the
# pose model is busy - the sound cuts out a couple of seconds in and stays out.
AUDIO_BUFFER = 512
AUDIO_BUFFER_LINUX = 2048
# Drop your own .ogg or .wav files in this folder and they are used instead
# of the generated music, one per world in name order. The folder is
# gitignored, so whatever you put there stays on your machine and is never
# published - which is the only safe way to use music you did not write.
MUSIC_DIR = os.path.join(ROOT_DIR, "music")
MUSIC_FILE_EXTS = (".ogg", ".wav")

# A single track here is cut into one loop per world when the game
# starts (well under a second). One small file in the project beats
# six big rendered ones, and it is the music everybody gets.
SOURCE_MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
SOURCE_MUSIC_EXTS = (".mp3", ".ogg", ".wav", ".flac")
# A file here whose name starts with this is the menu theme; anything
# else is the gameplay track that gets split per world.
MENU_MUSIC_PREFIX = "menu"

MUSIC_VOLUME = 0.34
# The menu theme sits well under the gameplay music on purpose - it
# loops the whole time you are on a menu, so it has to stay in the
# background rather than compete with it.
MENU_MUSIC_VOLUME = 0.15
SFX_VOLUME = 0.55
