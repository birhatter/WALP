"""WALP 2 - procedural audio.

Every sound and every music loop in the game is synthesised with numpy at
startup. There are no .wav or .ogg files to ship, nothing to download, and
nothing that can go missing.
"""

import threading
from typing import Optional

import numpy as np
import pygame

import config

SR = config.AUDIO_SAMPLE_RATE

# Semitone offsets for a major pentatonic scale, two octaves.
PENTATONIC = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24]
MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]


def note_hz(semitone: float) -> float:
    """Semitones relative to middle C (261.63 Hz)."""
    return 261.625565 * (2.0 ** (semitone / 12.0))


def _t(dur: float) -> np.ndarray:
    return np.arange(int(SR * dur), dtype=np.float32) / SR


def _decay(t: np.ndarray, dur: float, hardness: float = 3.5) -> np.ndarray:
    return np.exp(-t * (hardness / max(dur, 1e-4)))


def sine(freq: float, dur: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * _t(dur)).astype(np.float32)


def bell(freq: float, dur: float, amp: float = 1.0, bright: float = 1.0) -> np.ndarray:
    t = _t(dur)
    out = np.sin(2 * np.pi * freq * t)
    out += bright * 0.45 * np.sin(2 * np.pi * freq * 2.0 * t)
    out += bright * 0.22 * np.sin(2 * np.pi * freq * 3.01 * t)
    out += bright * 0.09 * np.sin(2 * np.pi * freq * 4.97 * t)
    return (amp * out * _decay(t, dur, 3.2)).astype(np.float32)


def pluck(freq: float, dur: float, amp: float = 1.0) -> np.ndarray:
    t = _t(dur)
    out = (np.sin(2 * np.pi * freq * t)
           + 0.32 * np.sin(2 * np.pi * freq * 2 * t)
           + 0.16 * np.sin(2 * np.pi * freq * 3 * t))
    return (amp * out * _decay(t, dur, 5.5)).astype(np.float32)


def pad(freq: float, dur: float, amp: float = 1.0) -> np.ndarray:
    """Soft sustained chord voice with a slow swell."""
    t = _t(dur)
    out = np.zeros_like(t)
    for k, w in ((1, 1.0), (2, 0.30), (3, 0.14), (4, 0.07)):
        out += w * np.sin(2 * np.pi * freq * k * t + k * 0.7)
    # gentle detune for warmth
    out += 0.5 * np.sin(2 * np.pi * freq * 1.005 * t)
    n = len(t)
    swell = np.minimum(1.0, np.linspace(0, 3.0, n)) * np.minimum(
        1.0, np.linspace(3.0, 0, n))
    return (amp * 0.35 * out * swell).astype(np.float32)


def chirp(f0: float, f1: float, dur: float, amp: float = 1.0) -> np.ndarray:
    t = _t(dur)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * max(dur, 1e-4)))
    return (amp * np.sin(phase)).astype(np.float32)


def noise(dur: float, amp: float = 1.0, smooth: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * dur)).astype(np.float32)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        x = np.convolve(x, kernel, mode="same").astype(np.float32)
        x *= np.sqrt(smooth)  # convolution lowers amplitude; put it back
    return amp * x


def _mix(dest: np.ndarray, src: np.ndarray, at_seconds: float) -> None:
    start = int(at_seconds * SR)
    if start >= len(dest) or start < 0:
        return
    end = min(len(dest), start + len(src))
    dest[start:end] += src[: end - start]


def _fade_edges(x: np.ndarray, ms: float = 6.0) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    if n * 2 >= len(x) or n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def _to_sound(mono: np.ndarray, gain: float = 0.9) -> pygame.mixer.Sound:
    mono = _fade_edges(np.asarray(mono, dtype=np.float32))
    peak = float(np.max(np.abs(mono))) if len(mono) else 1.0
    if peak > 1e-6:
        mono = mono / peak
    data = np.clip(mono * gain, -1.0, 1.0)
    ints = (data * 32767.0).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([ints, ints]))
    return pygame.sndarray.make_sound(stereo)


# --------------------------------------------------------------------------
# Sound effect recipes
# --------------------------------------------------------------------------

def _sfx_hit() -> np.ndarray:
    buf = np.zeros(int(SR * 0.85), dtype=np.float32)
    for i, semi in enumerate([0, 4, 7, 12]):
        _mix(buf, bell(note_hz(12 + semi), 0.55, amp=0.8 - i * 0.08), i * 0.055)
    _mix(buf, noise(0.10, amp=0.10, smooth=8, seed=1), 0.0)
    return buf


def _sfx_perfect() -> np.ndarray:
    buf = np.zeros(int(SR * 1.25), dtype=np.float32)
    for i, semi in enumerate([0, 4, 7, 12, 16, 19]):
        _mix(buf, bell(note_hz(12 + semi), 0.7, amp=0.75), i * 0.05)
    for i in range(7):
        _mix(buf, bell(note_hz(31 + PENTATONIC[i % 5]), 0.22, amp=0.28),
             0.34 + i * 0.045)
    return buf


def _sfx_crack() -> np.ndarray:
    buf = np.zeros(int(SR * 0.55), dtype=np.float32)
    _mix(buf, noise(0.14, amp=0.45, smooth=4, seed=7) * np.exp(
        -np.linspace(0, 6, int(SR * 0.14), dtype=np.float32)), 0.0)
    _mix(buf, pluck(note_hz(7), 0.22, amp=0.6), 0.05)
    _mix(buf, pluck(note_hz(14), 0.30, amp=0.6), 0.15)
    return buf


def _sfx_miss() -> np.ndarray:
    """Soft and cartoonish - never a harsh buzzer."""
    buf = np.zeros(int(SR * 0.75), dtype=np.float32)
    _mix(buf, chirp(note_hz(7), note_hz(0), 0.34, amp=0.5)
         * _decay(_t(0.34), 0.34, 2.2), 0.0)
    _mix(buf, chirp(note_hz(4), note_hz(-3), 0.36, amp=0.4)
         * _decay(_t(0.36), 0.36, 2.4), 0.18)
    return buf


def _sfx_heart() -> np.ndarray:
    buf = np.zeros(int(SR * 0.7), dtype=np.float32)
    _mix(buf, chirp(note_hz(12), note_hz(-5), 0.55, amp=0.55)
         * _decay(_t(0.55), 0.55, 2.8), 0.0)
    return buf


def _sfx_whoosh() -> np.ndarray:
    dur = 0.65
    t = _t(dur)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    body = noise(dur, amp=0.55, smooth=18, seed=11) * env
    body += 0.20 * chirp(180, 620, dur, amp=1.0) * env
    return body.astype(np.float32)


def _sfx_shatter() -> np.ndarray:
    buf = np.zeros(int(SR * 0.85), dtype=np.float32)
    burst = noise(0.30, amp=0.7, smooth=3, seed=23)
    burst *= np.exp(-np.linspace(0, 7, len(burst), dtype=np.float32))
    _mix(buf, burst, 0.0)
    rng = np.random.default_rng(5)
    for i in range(11):
        _mix(buf, pluck(note_hz(24 + rng.integers(0, 20)), 0.13, amp=0.20),
             0.06 + i * 0.05)
    return buf


def _sfx_tick() -> np.ndarray:
    return pluck(note_hz(24), 0.07, amp=0.5)


def _sfx_countdown() -> np.ndarray:
    return bell(note_hz(12), 0.28, amp=0.7)


def _sfx_go() -> np.ndarray:
    buf = np.zeros(int(SR * 0.6), dtype=np.float32)
    for i, semi in enumerate([0, 7, 12]):
        _mix(buf, bell(note_hz(19 + semi), 0.45, amp=0.8), i * 0.04)
    return buf


def _sfx_levelup() -> np.ndarray:
    buf = np.zeros(int(SR * 1.5), dtype=np.float32)
    for i, semi in enumerate([0, 2, 4, 7, 12, 16, 19, 24]):
        _mix(buf, bell(note_hz(7 + semi), 0.6, amp=0.7), i * 0.075)
    return buf


def _sfx_star() -> np.ndarray:
    buf = np.zeros(int(SR * 0.7), dtype=np.float32)
    for i, semi in enumerate([12, 19, 24]):
        _mix(buf, bell(note_hz(24 + semi), 0.35, amp=0.55), i * 0.06)
    return buf


def _sfx_pop() -> np.ndarray:
    return chirp(360, 900, 0.09, amp=0.55) * _decay(_t(0.09), 0.09, 3.0)


def _sfx_hover() -> np.ndarray:
    return pluck(note_hz(19), 0.10, amp=0.35)


def _sfx_gameover() -> np.ndarray:
    """Warm and encouraging, like the end of a cartoon - not a death sting."""
    buf = np.zeros(int(SR * 2.2), dtype=np.float32)
    for i, semi in enumerate([12, 9, 7, 4, 0]):
        _mix(buf, bell(note_hz(semi), 0.9, amp=0.62), i * 0.20)
    for i, semi in enumerate([0, 4, 7, 12]):
        _mix(buf, pad(note_hz(semi - 12), 1.1, amp=0.5), 1.05)
    return buf


def _sfx_unlock() -> np.ndarray:
    buf = np.zeros(int(SR * 1.1), dtype=np.float32)
    for i, semi in enumerate([0, 4, 7, 12, 19]):
        _mix(buf, bell(note_hz(19 + semi), 0.5, amp=0.7), i * 0.07)
    return buf


def _sfx_capture() -> np.ndarray:
    buf = np.zeros(int(SR * 0.5), dtype=np.float32)
    _mix(buf, noise(0.06, amp=0.4, smooth=2, seed=31), 0.0)
    _mix(buf, bell(note_hz(28), 0.35, amp=0.6), 0.02)
    return buf


def _sfx_combo(level: int) -> np.ndarray:
    """Pitch climbs with the combo so kids hear themselves doing well."""
    buf = np.zeros(int(SR * 0.5), dtype=np.float32)
    base = 12 + min(level, 8) * 2
    for i, semi in enumerate([0, 4, 7]):
        _mix(buf, bell(note_hz(base + semi), 0.3, amp=0.55), i * 0.035)
    return buf


def _sfx_pose_cue(index: int) -> np.ndarray:
    """A short motif unique to each pose - an audio 'name' for pre-readers."""
    rng = np.random.default_rng(1000 + index)
    buf = np.zeros(int(SR * 0.55), dtype=np.float32)
    steps = [int(rng.integers(0, 5)) for _ in range(3)]
    for i, s in enumerate(steps):
        _mix(buf, pluck(note_hz(19 + PENTATONIC[s]), 0.24, amp=0.5), i * 0.085)
    return buf


SFX_RECIPES = {
    "hit": _sfx_hit,
    "perfect": _sfx_perfect,
    "crack": _sfx_crack,
    "miss": _sfx_miss,
    "heart": _sfx_heart,
    "whoosh": _sfx_whoosh,
    "shatter": _sfx_shatter,
    "tick": _sfx_tick,
    "countdown": _sfx_countdown,
    "go": _sfx_go,
    "levelup": _sfx_levelup,
    "star": _sfx_star,
    "pop": _sfx_pop,
    "hover": _sfx_hover,
    "gameover": _sfx_gameover,
    "unlock": _sfx_unlock,
    "capture": _sfx_capture,
}


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------

def _choir(freq: float, dur: float, amp: float = 1.0) -> np.ndarray:
    """Breathy sustained voice. Three slightly detuned layers with a slow
    vibrato, which is what makes a held chord sound like people rather than
    an organ."""
    t = _t(dur)
    out = np.zeros_like(t)
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    for detune, weight in ((0.997, 0.9), (1.0, 1.0), (1.004, 0.9)):
        f = freq * detune * vib
        ph = 2 * np.pi * np.cumsum(f) / SR
        out += weight * (np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.10 * np.sin(3 * ph))
    n = len(t)
    swell = np.minimum(1.0, np.linspace(0, 2.2, n)) * np.minimum(1.0, np.linspace(2.6, 0, n))
    breath = noise(dur, amp=0.05, smooth=40, seed=int(freq) % 97)
    return (amp * 0.24 * (out * swell + breath * swell)).astype(np.float32)


def _taiko(dur: float = 0.30, amp: float = 0.7, seed: int = 2) -> np.ndarray:
    """Deep drum hit - the pulse the whole thing is built on."""
    t = _t(dur)
    body = (np.sin(2 * np.pi * 74 * t) + 0.5 * np.sin(2 * np.pi * 110 * t))
    body *= np.exp(-t * 11.0)
    skin = noise(dur, amp=0.35, smooth=3, seed=seed)
    skin *= np.exp(-np.linspace(0, 26, len(skin), dtype=np.float32))
    return (amp * (body + skin)).astype(np.float32)


def _swell(dur: float, amp: float = 0.5, seed: int = 4) -> np.ndarray:
    """Rising hiss that leans into the next phrase."""
    n = int(SR * dur)
    x = noise(dur, amp=amp, smooth=10, seed=seed)
    return (x * np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2.4).astype(np.float32)


# Phrygian-flavoured minor: the flattened second is what gives this kind of
# cue its dark, ancient colour.
DARK_SCALE = (0, 1, 3, 5, 7, 8, 10, 12)

# Four-note ostinato cells, as scale degrees. Short, insistent, and built to
# be repeated until they get under your skin.
OSTINATI = (
    (0, 1, 0, 4),
    (0, 3, 0, 1),
    (0, 4, 3, 1),
    (0, 1, 3, 4),
)


def build_music_loop(world: int) -> np.ndarray:
    """An original rising-tension loop.

    Written to do what a building orchestral cue does - a short ostinato that
    repeats and climbs, a low drone under it, choir holding the harmony and a
    drum pulse that thickens - rather than to imitate any particular piece.
    Later worlds get faster, higher and denser, so the music tightens as the
    run goes on.
    """
    theme = config.WORLDS[world % len(config.WORLDS)]
    root = theme["root"] - 12
    n_worlds = max(len(config.WORLDS) - 1, 1)
    intensity = (world % len(config.WORLDS)) / n_worlds      # 0 .. 1

    bpm = 104 + intensity * 30
    beat = 60.0 / bpm
    bar = beat * 4
    bars = 8
    total = bar * bars
    buf = np.zeros(int(SR * (total + 1.0)), dtype=np.float32)

    cell = OSTINATI[world % len(OSTINATI)]
    step = beat * 0.5                       # the ostinato runs in eighths
    # chords under it: i - i - VI - VII, the classic climbing minor shape
    chords = (0, 0, 8, 10, 0, 0, 8, 10)

    for b in range(bars):
        t0 = b * bar
        chord = root + chords[b % len(chords)]
        phrase = b // 4                     # second half sits an octave up
        climb = 12 * phrase

        # ---- drone ----
        _mix(buf, _choir(note_hz(root - 12), bar * 1.02, amp=0.55 + 0.2 * intensity), t0)

        # ---- choir chord: root, minor third, fifth ----
        for semi in (0, 3, 7):
            _mix(buf, _choir(note_hz(chord + semi), bar * 0.98,
                             amp=0.30 + 0.15 * intensity), t0)

        # ---- the ostinato: the engine of the whole thing ----
        for i in range(8):
            deg = cell[i % len(cell)]
            semi = chord + DARK_SCALE[deg % len(DARK_SCALE)] + 12 + climb
            accent = 1.25 if i % 4 == 0 else 1.0
            _mix(buf, pluck(note_hz(semi), step * 0.92,
                            amp=(0.20 + 0.14 * intensity) * accent),
                 t0 + i * step)
            if intensity > 0.35:            # doubled an octave up when tense
                _mix(buf, pluck(note_hz(semi + 12), step * 0.55, amp=0.09),
                     t0 + i * step)

        # ---- bass on the strong beats ----
        _mix(buf, pluck(note_hz(chord - 12), beat * 1.4, amp=0.42), t0)
        _mix(buf, pluck(note_hz(chord - 12), beat * 0.8, amp=0.30), t0 + beat * 2)

        # ---- drums ----
        _mix(buf, _taiko(amp=0.62, seed=world * 3), t0)
        _mix(buf, _taiko(amp=0.44, seed=world * 3 + 1), t0 + beat * 2)
        if intensity > 0.2:
            _mix(buf, _taiko(amp=0.30, seed=world * 3 + 2), t0 + beat * 3.5)
        if intensity > 0.6:
            for k in (1, 3):
                _mix(buf, _taiko(dur=0.16, amp=0.22, seed=world + k), t0 + k * beat)

        # ---- swell into each four-bar phrase ----
        if b % 4 == 3:
            _mix(buf, _swell(beat * 2.0, amp=0.16 + 0.14 * intensity, seed=world + 9),
                 t0 + beat * 2)

    buf = buf[: int(SR * total)]
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = np.tanh(buf / peak * 1.6) / np.tanh(1.6)
    return (buf * 0.84).astype(np.float32)


# --------------------------------------------------------------------------
# Splitting one supplied track into per-world loops
# --------------------------------------------------------------------------
SPLIT_WINDOW = 0.05      # envelope resolution, seconds
SPLIT_SEARCH = 5.0       # how far either side of an even split to hunt for a lull
SPLIT_FADE = 0.40        # crossfade baked into each loop point, seconds
SPLIT_GUARD = 3.0        # never cut this close to either end


def _envelope(audio: np.ndarray) -> np.ndarray:
    mono = audio.mean(axis=1)
    n = int(SR * SPLIT_WINDOW)
    usable = (len(mono) // n) * n
    if usable < n:
        return np.zeros(1, dtype=np.float32)
    return np.sqrt((mono[:usable].reshape(-1, n) ** 2).mean(axis=1))


def _find_cuts(env: np.ndarray, pieces: int) -> list:
    """Even splits, pulled onto the quietest window nearby, so a piece never
    starts or ends in the middle of a phrase."""
    total = len(env)
    guard = int(SPLIT_GUARD / SPLIT_WINDOW)
    span = int(SPLIT_SEARCH / SPLIT_WINDOW)
    cuts = []
    for i in range(1, pieces):
        ideal = int(total * i / pieces)
        lo, hi = max(guard, ideal - span), min(total - guard, ideal + span)
        if hi <= lo:
            cuts.append(ideal)
            continue
        window = env[lo:hi]
        bias = np.abs(np.arange(lo, hi) - ideal) / max(span, 1) * env.mean() * 0.35
        cuts.append(lo + int(np.argmin(window + bias)))
    return cuts


def _loopable(seg: np.ndarray) -> np.ndarray:
    """Crossfade the tail into the head so a piece repeats without a click."""
    fade = int(SR * SPLIT_FADE)
    if len(seg) <= fade * 3:
        return seg
    head, tail = seg[:fade].copy(), seg[-fade:]
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    out = seg[:-fade].copy()
    out[:fade] = head * ramp + tail * (1.0 - ramp)
    return out


def split_track(path: str, pieces: int) -> list:
    """Cut one track into `pieces` loopable parts, quietest first.

    Ordered by loudness rather than by timestamp: the game gets harder world
    by world, so the music should climb with it.
    """
    snd = pygame.mixer.Sound(path)
    audio = pygame.sndarray.array(snd).astype(np.float32) / 32768.0
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    env = _envelope(audio)
    step = int(SR * SPLIT_WINDOW)
    cuts = _find_cuts(env, pieces)
    bounds = [0] + [c * step for c in cuts] + [len(audio)]

    scored = []
    for i in range(pieces):
        chunk = env[bounds[i] // step: bounds[i + 1] // step]
        level = float(chunk.mean()) if len(chunk) else 0.0
        scored.append((level, bounds[i], bounds[i + 1]))
    scored.sort(key=lambda x: x[0])
    return [_loopable(audio[lo:hi]) for _, lo, hi in scored]


def _find_source_tracks():
    """(gameplay track, menu track) from assets/music. Either may be None."""
    import os
    try:
        names = sorted(os.listdir(config.SOURCE_MUSIC_DIR))
    except OSError:
        return None, None
    game = menu = None
    for name in names:
        if not name.lower().endswith(config.SOURCE_MUSIC_EXTS):
            continue
        path = os.path.join(config.SOURCE_MUSIC_DIR, name)
        if name.lower().startswith(config.MENU_MUSIC_PREFIX):
            menu = menu or path
        else:
            game = game or path
    return game, menu


def load_loop(path: str) -> Optional[pygame.mixer.Sound]:
    """Load a track for looping, smoothing the seam only if it needs it."""
    try:
        snd = pygame.mixer.Sound(path)
        audio = pygame.sndarray.array(snd).astype(np.float32) / 32768.0
    except Exception:
        return None
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    # A track written to loop already joins cleanly; only crossfade one that
    # would otherwise click, so a deliberate loop is left untouched.
    if float(np.abs(audio[0] - audio[-1]).mean()) > 0.02:
        audio = _loopable(audio)
    ints = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(ints))


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class Audio:
    def __init__(self) -> None:
        self.ok = False
        self.sfx = {}
        self.combo_sfx = []
        self.pose_cues = []
        self._music = {}
        self._music_lock = threading.Lock()
        self._current_world = None
        self._music_channel = None
        self.custom_music = []
        self.menu_music = None
        self._current_kind = None
        self.muted = False

    # -- setup ------------------------------------------------------------
    def init(self) -> None:
        try:
            pygame.mixer.pre_init(SR, -16, 2, 512)
            pygame.mixer.init(SR, -16, 2, 512)
            pygame.mixer.set_num_channels(32)
            pygame.mixer.set_reserved(1)
            self._music_channel = pygame.mixer.Channel(0)
        except pygame.error:
            self.ok = False
            return

        try:
            for name, recipe in SFX_RECIPES.items():
                snd = _to_sound(recipe())
                snd.set_volume(config.SFX_VOLUME)
                self.sfx[name] = snd
            for lvl in range(9):
                snd = _to_sound(_sfx_combo(lvl))
                snd.set_volume(config.SFX_VOLUME)
                self.combo_sfx.append(snd)
            for idx in range(24):
                snd = _to_sound(_sfx_pose_cue(idx))
                snd.set_volume(config.SFX_VOLUME * 0.8)
                self.pose_cues.append(snd)
        except Exception:
            self.ok = False
            return

        self._load_music_files()
        self.ok = True
        if self.custom_music:
            return

        # World 0's loop is needed immediately; the rest are built in the
        # background so startup stays snappy.
        self._ensure_music(0, blocking=True)
        threading.Thread(target=self._prebuild_music, daemon=True).start()

    def _prebuild_music(self) -> None:
        for w in range(1, len(config.WORLDS)):
            self._ensure_music(w, blocking=True)

    def _load_music_files(self) -> None:
        """Use the player's own tracks from config.MUSIC_DIR if there are any.

        Kept as an opt-in folder rather than shipped audio: music you did not
        write cannot go in the repository, but there is no reason you cannot
        play your own copy locally.
        """
        import os
        self.custom_music = []
        try:
            names = sorted(os.listdir(config.MUSIC_DIR))
        except OSError:
            # No override folder is the normal case - fall through to the
            # supplied track below rather than giving up here.
            names = []
        for name in names:
            if not name.lower().endswith(config.MUSIC_FILE_EXTS):
                continue
            try:
                snd = pygame.mixer.Sound(os.path.join(config.MUSIC_DIR, name))
            except (pygame.error, OSError):
                continue
            self.custom_music.append(snd)
        if self.custom_music:
            print(f"Using {len(self.custom_music)} track(s) from {config.MUSIC_DIR}")
            return

        # Nothing pre-split? Take the supplied track and cut it up now. It
        # costs well under a second and keeps one small source file in the
        # project instead of six large rendered ones.
        source, menu_path = _find_source_tracks()
        if menu_path:
            self.menu_music = load_loop(menu_path)
            if self.menu_music:
                import os
                print(f"Menu theme: {os.path.basename(menu_path)}")
        if source is None:
            return
        try:
            parts = split_track(source, len(config.WORLDS))
        except Exception as exc:
            print(f"Could not split {source}: {exc}")
            return
        for part in parts:
            ints = (np.clip(part, -1.0, 1.0) * 32767.0).astype(np.int16)
            self.custom_music.append(
                pygame.sndarray.make_sound(np.ascontiguousarray(ints)))
        import os
        print(f"Split {os.path.basename(source)} into "
              f"{len(self.custom_music)} world loops")

    def _ensure_music(self, world: int, blocking: bool = False) -> bool:
        world = world % len(config.WORLDS)
        if self.custom_music:
            return True
        with self._music_lock:
            if world in self._music:
                return True
        if not blocking:
            return False
        try:
            arr = build_music_loop(world)
            snd = _to_sound(arr, gain=0.8)
        except Exception:
            return False
        with self._music_lock:
            self._music[world] = snd
        return True

    # -- playback ---------------------------------------------------------
    def play(self, name: str, volume: float = 1.0) -> None:
        if not self.ok or self.muted:
            return
        snd = self.sfx.get(name)
        if snd is None:
            return
        ch = pygame.mixer.find_channel(True)
        if ch is not None:
            ch.set_volume(min(1.0, config.SFX_VOLUME * volume))
            ch.play(snd)

    def play_combo(self, level: int) -> None:
        if not self.ok or self.muted or not self.combo_sfx:
            return
        snd = self.combo_sfx[min(level, len(self.combo_sfx) - 1)]
        ch = pygame.mixer.find_channel(True)
        if ch is not None:
            ch.play(snd)

    def play_pose_cue(self, index: int) -> None:
        if not self.ok or self.muted or not self.pose_cues:
            return
        snd = self.pose_cues[index % len(self.pose_cues)]
        ch = pygame.mixer.find_channel(True)
        if ch is not None:
            ch.play(snd)

    def play_menu(self) -> None:
        """Loop the menu theme, quietly, until gameplay takes over."""
        if not self.ok or self.menu_music is None or self._music_channel is None:
            return
        if self._current_kind == "menu":
            return
        self._current_kind = "menu"
        self._current_world = None
        self._music_channel.stop()
        self._music_channel.set_volume(
            0.0 if self.muted else config.MENU_MUSIC_VOLUME)
        self._music_channel.play(self.menu_music, loops=-1, fade_ms=800)

    def set_world(self, world: int) -> None:
        if not self.ok:
            return
        world = world % len(config.WORLDS)
        if world == self._current_world and self._current_kind == "world":
            return
        if not self._ensure_music(world):
            return  # still baking; keep whatever is playing
        if self.custom_music:
            snd = self.custom_music[world % len(self.custom_music)]
        else:
            with self._music_lock:
                snd = self._music.get(world)
        if snd is None:
            return
        self._current_world = world
        self._current_kind = "world"
        if self._music_channel is None:
            return
        self._music_channel.stop()
        self._music_channel.set_volume(0.0 if self.muted else config.MUSIC_VOLUME)
        self._music_channel.play(snd, loops=-1, fade_ms=700)

    def set_music_volume(self, scale: float) -> None:
        if self.ok and self._music_channel is not None:
            base = (config.MENU_MUSIC_VOLUME if self._current_kind == "menu"
                    else config.MUSIC_VOLUME)
            self._music_channel.set_volume(0.0 if self.muted else base * scale)

    def stop_music(self) -> None:
        if self.ok and self._music_channel is not None:
            self._music_channel.fadeout(400)
            self._current_world = None

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        self.set_music_volume(1.0)
        return self.muted

    def shutdown(self) -> None:
        if self.ok:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except pygame.error:
                pass
