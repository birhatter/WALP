# WALP

A webcam game. A wall rushes at you with a hole cut in the shape of a person.
Match that shape with your body and you go straight through it.

No controller, no mouse, no keyboard — just stand in front of the camera.

## Running it

Double-click **`play_walp2.bat`**, or:

```bash
python walp2.py
```

| Option | What it does |
|---|---|
| `--windowed` | run in a window instead of fullscreen |
| `--list-cameras` | show which camera indexes work, then exit |
| `--camera 1` | use a specific camera (remembered afterwards) |
| `--dshow` | force the DirectShow backend (starts faster, slower on some webcams) |
| `--selftest` | run the developer self test (needs `dev/`, see below) |

Keys: `ESC` pause · `R` record (off by default) · `M` mute · `F3` stats · `Ctrl+Q` quit.

If the hand pointer misbehaves, press `F3` — the overlay shows whether a
body is seen and where the pointer thinks your hand is.

## Using an external webcam

Plug it in, then run:

```bash
python walp2.py --list-cameras
```

It prints every index that works. The external camera is almost always the
**highest** number. Then start with that index once:

```bash
python walp2.py --camera 1
```

The choice is saved to `walp2_save.json`, so from then on plain
`python walp2.py` keeps using it. To go back to the built-in camera, run it
once with `--camera 0`.

If the external camera is slow to start, add `--dshow`.

## How to play

1. **Get your head, shoulders and hips in frame.** Your legs do *not* have to
   fit — if they are out of shot the game simply never asks for a squat or a
   kick. Small rooms are fine. The camera view is bottom-left so you can see
   yourself while you line up, and `ENTER` skips this screen.
2. On the menu, **raise one hand above your shoulders** — that becomes a
   pointer. Move it left and right and hold for a moment to select.

   - The hand must clear your shoulders. An arm at rest is ignored on
     purpose, so nothing gets picked by accident.
   - Selection **snaps to the nearest button**, so pointing roughly at one is
     enough — you do not have to be accurate.
   - It is measured against your own shoulder width, so the whole row is
     reachable whether you stand close or right across the room.
   - Arrow keys and `ENTER` always work as a fallback, and `1`–`5` jump
     straight to a button.
3. In the game, **match the shape in the wall** before it reaches you.

**On-screen help while playing**

- **Bottom-left** — live camera, so you can see where you are standing.
- **Bottom-right** — the target shape, with *your* limbs coloured in: green
  where you match, red where you do not. This is the fastest way to see which
  arm or leg is wrong.
- **Right** — a live match percentage. The ring fills as you get closer, and
  the small arrow on the ring is the pass mark.

## The rules

- **One number decides everything.** The percentage on the right *is* the
  test. If it reads past the arrow, you pass. There is no hidden second check.
- **It never asks the impossible.** If the camera cannot see your legs, you
  will never be dealt a squat or a kick.
- **Near misses count.** Early on, getting most of the way there scrapes you
  through for half points and costs no life. That safety net shrinks as you
  go, and is gone by wall 45.
- **Mirroring is fine.** Raising the other arm still counts.
- **Easy shapes first.** Harder poses unlock as you clear walls.

## It always ends

The difficulty never levels off, so every run finishes:

| Wall | Time to react | Judging window | Tolerance | Near-miss rescue |
|-----:|--------------:|---------------:|----------:|-----------------:|
| 1    | 5.2 s         | 2.34 s         | very kind | on |
| 10   | 3.1 s         | 0.93 s         | kind      | on |
| 20   | 2.0 s         | 0.42 s         | normal    | fading |
| 40   | 1.15 s        | 0.14 s         | tight     | almost gone |
| 80   | 0.6 s         | 0.05 s         | brutal    | off |
| 110  | 0.35 s        | 0.03 s         | brutal    | off |

**Three lives**, and you get one back every 10 walls cleared (never more than
three at once). The walls speed up far faster than the lives come back.

## Recording

**Currently switched off.** `RECORDING_ENABLED = False` in `config.py`.
Nothing is recorded, no folder is made and the `R` key does nothing. Set it
back to `True` to re-enable — the whole feature is intact.

When enabled, the game saves a video of each attempt to `recordings/` (created when the
game starts, next to `walp2.py`), one file per attempt named
`run_YYYYmmdd_HHMMSS.mp4`.

Recording starts the moment a run begins — after you pick something from the
menu and the screen transition finishes, never on the menu itself. It is
closed and written to disk when the run ends, **and also if you leave the run
early** through pause or the menu. A video file is only playable once it has
been closed, so this matters: abandoning a run used to leave an unfinished
file behind.

The path of each saved clip is printed to the console and shown on the
results screen. Press **`R`** at any time to start or stop a clip
manually. A blinking **REC** dot in the top-right shows when it is running.

Encoding happens on a background thread with a small buffer, so it costs very
little speed (~130 fps down to ~107 in testing) and drops frames rather than
stuttering the game. Clips are 1280x720 at 30 fps.

To turn it off, set `RECORD_AUTO_RUNS = False` in `config.py` (or
`RECORDING_ENABLED = False` to disable it entirely).

**Be aware:** the recording is of the whole screen, which includes the live
camera view — so the files contain video of whoever is playing. They stay in
`recordings/` on this machine and nothing uploads or shares them. They are
ordinary video files, so anyone with access to the machine can open them.
Delete the folder whenever you like.

`recordings/` is in `.gitignore`, along with every common video extension, so
clips can never be committed or pushed to GitHub by accident.

The only other files written are `walp2_save.json` (best score, chosen camera,
settings) and `my_poses.json`.

## Files

| File | What it is |
|---|---|
| `walp2.py` | Game loop, modes, difficulty, scoring |
| `config.py` | Every tunable number, in one place |
| `poses.py` | Pose library, scoring, which poses are fair to ask |
| `wall.py` | The wall, the hole, the shatter |
| `visuals.py` | Fonts, worlds, particles, HUD, camera view, hand cursor |
| `vision.py` | Camera + tracking thread, calibration |
| `audio.py` | All music and sound, synthesised at startup |
| `recording.py` | Background screen recorder |
| `assets/` | The MediaPipe pose models and the music track |

Those nine files plus `assets/` are the whole game. Development tools live in
`dev/` and are deliberately **not** published — they are not needed to play,
and shipping test rigs to players only invites confusion:

| `dev/` (local only) | What it is |
|---|---|
| `selftest.py` | Headless run of every screen, a full playthrough, the difficulty curve and the menu pointer |
| `tools_preview.py` | Renders preview PNGs without a camera |
| `tools_handcheck.py` | Try the menu pointer on its own |
| `tools_looks.py` | Render alternative visual styles side by side |
| `tools_music.py` | Export the music loops as .wav to listen to |

There are no sound or image assets — every sound is generated with numpy at
startup and every graphic is drawn from shapes, so nothing can go missing.

## Music

The soundtrack comes from one track that the game splits per world.

**The supplied tracks.** `assets/music/` holds two files: the menu theme
(any filename starting with `menu`) and the gameplay track (anything else).

The menu theme loops continuously and quietly behind every menu, calibration
and results screen — it sits at 44% of the gameplay volume so it stays in the
background rather than competing.

The gameplay track is the one that gets split. On startup the
game cuts it into one loop per world — the cuts land on the quietest moment
near each even division so a piece never starts mid-phrase, each piece is
crossfaded end-into-start so it loops without a click, and the pieces are
handed out quietest-first so the music climbs as the game gets harder. The
whole thing takes about half a second.

Drop in different files (`.mp3`, `.wav`, `.ogg`, `.flac`) and they are used
instead — no code change. Name the menu one `menu*`. A `.wav` master gives the best result; converting an
mp3 to wav first gains nothing, because the loss already happened.

**Overriding per world.** If you want to hand-pick each world's music, put
`.ogg`/`.wav` files in a `music/` folder at the top level. Those win over the
supplied track. That folder is gitignored, so it stays on your machine.

**On licensing:** only publish music you have the right to publish. Buying a
track usually licenses you to *use* it, which is not always the same as
redistributing it inside a public repository — worth checking the licence you
bought. If in doubt, keep it in the local `music/` folder instead of
`assets/music/`, and the game falls back to its own generated soundtrack for
anyone who clones the project.

**No music file at all?** The game generates its own: an original
rising-tension loop per world — a short ostinato in a phrygian minor that
repeats and climbs, a low drone, a choir pad and a drum pulse.

## Tuning

Open `config.py`:

- **Too slow?** Lower `TRAVEL_START` and `SPAWN_GAP_SECONDS`.
- **Too fast?** Raise `TRAVEL_START`, raise `TOLERANCE_SCALE_EASY`.
- **Too hard?** Lower `PASS_THRESHOLD`, raise `START_LIVES`.
- **Ends too soon?** Raise `TRAVEL_MIN` / `TRAVEL_HARD_FLOOR` — but making
  these too generous is exactly what would let a run go on forever.
- **Music too loud or quiet?** `MUSIC_VOLUME` for gameplay,
  `MENU_MUSIC_VOLUME` for the menu theme.
- **Different look?** `WORLDS` holds the six themes; `FONT_PREFERENCES` in
  `visuals.py` is empty so it uses pygame's built-in face — put a font name in
  that list to change it.
- **Hole shape:** `LIMB_BODY_UNITS` is the limb thickness in body units. These
  are body-relative, not wall-relative, which is what keeps a squat looking
  like a person instead of a blob.

## Putting this on GitHub

`.gitignore` is written on the principle that **if it is not needed to run the
game on someone else's machine, it is not published.** It excludes:

- **Secrets** — `.env`, `*.key`, `*.pem`, `credentials*`, `*token*`, `api_key*`,
  `.aws/`, `.ssh/`, `id_rsa*`, and AI-assistant folders like `.claude/`,
  `.cursor/`, `.copilot*` that can carry keys or history
- **Recordings** — `recordings/` and every video extension; these are footage
  of the player
- **Developer tools** — `dev/`, `preview/`, `music_preview/`, `*.wav`
- **Local state** — `walp2_save.json`, `my_poses.json`
- **Editor settings** — `.vscode/`, `.idea/`, `*.code-workspace`
- Python caches, build output and OS junk

The two model files in `assets/` (~15 MB together) **are** tracked, so a fresh
clone runs straight away.

Before your first push, check what is actually staged:

```bash
git add -A && git status --short
```

You should see exactly thirteen files: `.gitignore`, `README.md`,
`play_walp2.bat`, the nine `.py` files, and the two `assets/*.task` models.
Anything else, stop and add it to `.gitignore` first.

## Requirements

Python **3.11** (3.9–3.12 are fine; **mediapipe does not support 3.13 yet**),
with `pygame`, `opencv-python`, `mediapipe`, `numpy`.

```bash
python -m pip install pygame opencv-python mediapipe numpy
```

## `No module named 'pygame'`

This machine has several Pythons installed and only 3.11 has the libraries.
If you see that error you are running the wrong one — check the path printed
in the traceback: `python3.13.exe` means the wrong interpreter, not a missing
package.

- **Easiest:** launch with `play_walp2.bat`. It tries each Python it can find
  and uses the first one that can import everything.
- **In VS Code:** press `Ctrl+Shift+P`, choose *Python: Select Interpreter*,
  and pick **Python 3.11**. This folder's `.vscode/settings.json` already
  points at it, so reopening the folder usually fixes it too.
- **In a terminal:** call it directly.

  ```bash
  python3.11 walp2.py
  ```

## Licence

WALP is free to play but it is **not** open source.

You may download it, play it, and record or stream yourself playing it. You
may not redistribute it, modify it, publish it elsewhere, or use it
commercially without permission. The music is the author's own work and is
licensed only as part of this game.

Full terms: [LICENCE.md](LICENCE.md) · Third-party components:
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)

Contributions are not being accepted. If you want to do something the licence
does not allow, open an issue and ask.
