# Handy IO

## Goal
Build a computer vision prototype for natural screen interaction using eyes, face, and hands.
The core idea: track the user's eyes, face, and hands in 3D, then infer what screen region lies
just behind each fingertip (or between both hands) and let the user interact with on-screen objects
through natural gestures — no mouse or keyboard needed.

## Milestones
1. **Gaze heatmap** — webcam input, detect face + eye landmarks, project estimated gaze onto screen,
   render a live heatmap overlay showing where the user is probably looking.
2. **Hand tracking** — detect hands and finger landmarks in 3D; overlay skeleton on a 2D UI.
3. **Gaze-behind-finger** — combine eye gaze ray with fingertip 3D position to identify the screen
   object that sits just behind the fingertip from the user's perspective.
4. **Gesture interaction** — map hand gestures (pinch, point, swipe, two-hand spread) to UI actions
   (select, drag, resize, dismiss) on 2D form-based widgets.

## Architecture — preferred module split
```
src/handy_io/
  capture.py        # webcam capture loop (OpenCV)
  landmarks.py      # MediaPipe face-mesh + hand-landmark inference
  gaze.py           # eye-to-screen gaze estimation
  heatmap.py        # gaussian accumulation + colormap overlay
  gestures.py       # gesture classifier on hand landmarks
  ui.py             # 2D widget layer (pygame or tkinter)
  main.py           # entry point wiring all modules
cpp/
  CMakeLists.txt    # optional native acceleration later
```

## Stack
- **Python** — prototyping, CV pipeline, UI experiments.
- **C++/CMake** — later native acceleration or optimized vision modules (not the starting point).
- **MediaPipe** — face mesh (468 landmarks) + hand tracking (21 landmarks per hand).
- **OpenCV** — webcam capture, image ops, overlay rendering.
- **NumPy** — landmark math, heatmap accumulation.
- **pygame or tkinter** — 2D UI for the first interactive demos.

## Environment
- Use `uv` to manage the virtual environment (`.venv/`).
- `uv sync --extra dev` installs all deps including pytest.
- Python ≥ 3.11 required.

## Conventions
- Keep modules small and single-responsibility.
- No global state — pass config/state explicitly.
- Prefer modular components: camera capture, landmark detection, UI rendering are separate concerns.
- First prototype: webcam input → gaze overlay → basic interaction hooks.
- Keep it simple before adding 3D depth or ML refinement.
