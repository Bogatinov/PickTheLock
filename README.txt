PICK THE LOCK BOT
=================

An educational project for learning real-time computer vision in Python:
screen capture through DXGI desktop duplication, image analysis with OpenCV
and NumPy, object tracking, and latency compensation. The subject is the
"Pick the Lock" minigame from the Dota 2 Dark Carnival event: a dial with a
rotating pick and shrinking target bars makes a compact, fast-moving test
case for these techniques.

The program only looks at the screen: the dial is captured with DXGI
desktop duplication, the pick and the yellow/blue bars are found with
OpenCV, and a normal mouse click is sent when the pick will be on a bar at
the moment the game registers it. Nothing hooks into Dota or Steam and no
game memory is read or modified.

Topics covered in the code:
  - DXGI desktop duplication (via dxcam) with an mss fallback
  - polar unwrapping of a circular region (cv2.warpPolar)
  - background modelling and motion detection
  - colour segmentation in HSV, robust to fading/translucent objects
  - template-matching OCR for an on-screen timer
  - tracking with outlier rejection and velocity estimation
  - learning input-to-display latency from visual feedback
  - a small look-ahead planner, tuned with Monte Carlo simulation


DISCLAIMER
----------
This repository is provided for educational purposes only, to explore
Python, DXGI and OpenCV. It is not intended to give anyone an unfair
advantage, and you are responsible for how you use it. Automating gameplay
may break the Dota 2 / Steam terms of service; use it at your own risk.

This project is not affiliated with, endorsed or sponsored by Valve
Corporation. Dota 2 and Steam are trademarks of Valve Corporation. No game
files are included. The only game-derived data are tiny binarized digit
templates (16x24 pixels) used to read the on-screen timer; the image in
docs\ is drawn by the project's own simulator.


REQUIREMENTS
------------
- Windows 10/11
- Python 3.13 (only to run from source or to build the exe)
- Dota 2 in Borderless Window mode (recommended)


RUNNING FROM SOURCE
-------------------
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python picklock.py

BUILDING THE EXE
----------------
    build.bat

PickTheLock.exe is written to the repository root, next to config.json.


FIRST-TIME CALIBRATION
----------------------
Open the minigame, then (see docs\calibration_guide.png):

  1. Hover the mouse over the centre of the lock dial      -> press F5
  2. Hover over the outer edge of the dark ring, where a
     yellow bar ends                                       -> press F6

You don't click, you only hover and press the key. The result is stored in
config.json and reused on every start. Recalibrate if the window size or
resolution changes.


HOTKEYS
-------
  F5    calibrate the centre
  F6    calibrate the ring edge
  F7    start / stop (high beep = on, low beep = off)
  F8    debug view + recording on / off
  DEL   quit

The bot only clicks while the Dota 2 window is in the foreground. It holds
the right mouse button (speed boost) while it plays.


HOW IT PLAYS
------------
- Tracks the pick and predicts where it will be when a click lands. The
  click-to-game delay is learned from the hit/miss flash after each click.
- Before each click it looks ~2.5 s ahead and may let a bar pass to reach a
  better one. Blue bars score like yellow ones and add 1.5 s to the clock,
  and more bars appear the longer a game lasts, so time is worth a lot.
- Reads the TIME REMAINING display and ignores bars it can't reach in time.
- Hit or miss is read from the flash colour (blue-white / red). After a miss
  it pauses for 600 ms, since the game disables the pick anyway.


FRAME RATE
----------
The bot sees the minigame as often as Dota redraws it (fps_max_ui).
"fps_profile" in config.json selects the timing profile:

  "auto"  default, picks one of the two below from the measured rate
  "120"   fps_max_ui 120 or higher, most precise
  "30"    fps_max_ui 30, slightly fewer hits per game

Below ~25 fps the hit flash is often not drawn at all; avoid that.


CONFIGURATION
-------------
config.json holds the calibration, the learned latency per profile and any
setting you want to override. Only add what you want to change, e.g.

    { "fps_profile": "30", "min_target_s": 0.03 }

Useful settings (all defaults are in DEFAULTS at the top of picklock.py):

  fps_profile         "auto" | "120" | "30"
  boost_mode          "always" | "off"
  strategy            "plan" | "greedy" (click every bar it reaches)
  points_per_second   value of the time a blue bar adds (default 1600)
  min_target_s        skip bars the pick crosses faster than this (0.025);
                      raise it if you see misses on thin bars
  click_blue          false to ignore blue bars
  yellow_hue,
  blue_hue            colour ranges, if bars aren't detected
                      (console shows bars(y/b)=0/0)


CONSOLE STATUS LINE
-------------------
  [ON ] fps=164.8/120 pick= 182.3 w=  127 bars(y/b)=2/1 lat=21.4ms timer=17.3 ...

  fps      frames per second / active profile
  pick     pick angle, w = its speed in deg/s (~127 with boost)
  bars     yellow / blue bars currently detected
  lat      learned click-to-game delay
  timer    TIME REMAINING as read from the screen


RECORDINGS AND DEVELOPER TOOLS
------------------------------
F8 writes recordings\rec_<time>.mkv (lossless, ~1 GB per game), a .csv
with what the bot detected and did on every frame, and a .json with the
settings in use. Run the tools from the repository root:

  python -m tools.replay recordings\rec_<time>
      Replays a recording through the current code and saves an annotated
      frame for every click it would make.

  python -m tools.strategy_sim [games] [fps]
      Plays whole games against a model of the minigame fitted to real
      recordings. Use it to compare strategy settings before trying them
      in the game (60+ games, single games vary a lot).

  python -m tools.vision_sim [seconds]
      Renders a synthetic dial and runs the full detection + control
      pipeline on it, at 165 and 30 fps.


PROJECT LAYOUT
--------------
  picklock.py           the bot (single file)
  tools\                replay and simulators
  docs\                 calibration picture
  build.bat             PyInstaller build
  requirements.txt      pinned dependencies
  LICENSE               MIT license


TROUBLESHOOTING
---------------
- Nothing happens on F7: listen for the beep. A low beep means F7 just
  switched the bot off. It also only clicks while Dota is focused.
- bars(y/b)=0/0 while bars are visible: recalibrate (F5/F6).
- Misses on thin bars: raise min_target_s, e.g. 0.03.
- fps far below your fps_max_ui: close other screen recorders/overlays.


LICENSE
-------
MIT, see LICENSE. Provided as is, without warranty of any kind.
