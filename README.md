# Rock–Paper–Scissors Gesture Recognition

IKT213 Machine Vision — Group 13

A webcam plays rock, paper, scissors against you. The system tracks 21 joints
of the hand in real time and is designed to decide which gesture is being
formed while the hand is still moving.

The hand tracking is our own: a detector and a landmark model we trained
ourselves, following the same two-stage architecture MediaPipe Hands uses.
MediaPipe is our reference baseline and our labelling tool, not a component of
the system.

**This is a first prototype.** Hand tracking works and is measurable. Gesture
classification and the game itself are not built yet — see section 3.

---

## 1. Installation

Requirements:

- Windows 11
- Python 3.12
- A webcam
- NVIDIA GPU with CUDA 12.x — optional, but gives ~12 ms per frame instead of
  running on the CPU

Check your Python version first:

```bash
python --version
```

Clone and create a virtual environment:

```bash
git clone https://github.com/MagnusGreen123/ikt213_prosjekt.git
cd ikt213_prosjekt
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`.

**Install PyTorch before anything else.** It is deliberately not in
`requirements.txt`, because it has to come from its own package index to get
GPU support.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Without an NVIDIA GPU, drop the `--index-url` part. Tested with
torch 2.11.0+cu128 and torchvision 0.26.0+cu128.

Then the rest:

```bash
pip install -r requirements.txt
```

The trained model weights are committed to the repository, so there is nothing
to download. After cloning, `models/` should contain `detector_palm.pt` and
`finetuned_palm_frei_v3_70.pt`.

---

## 2. Running

Start with the live viewer:

```bash
python scripts/live_pipeline_v2.py
```

You should see your camera feed with a hand skeleton drawn on it and timing
figures in the top-left corner.

| Key | Action |
|---|---|
| `q` | Quit |
| `d` | Toggle tracking on/off |
| `r` | Reset tracking state |

**For the best result:** hold your hand side-on to the camera, roughly 50 cm
away, against a dark background, in a well lit room. Section 3 explains why.

### Diagnostics

```bash
python scripts/diagnose_full_v2.py
```

Nine scripted scenarios, twice each, eight seconds per run. It prompts on
screen before each one — press **space** to start a scenario or **q** to skip
it. Results are written to `logs/diagnose_full.json` and a summary table is
printed at the end. This is the script that produces the measurements in the
test plan.

### Recording new data

```bash
python scripts/collect_data.py
```

Records labelled throws to `data/recordings/`. Not needed to test the system.

---

## 3. What works, and what does not yet

Implemented and measurable in this prototype:

- Hand detection, including correctly reporting that no hand is present
- 21-joint tracking at ~12 ms per frame
- Live visualisation of the tracked skeleton
- Diagnostics against MediaPipe as a reference

Not implemented yet:

- Gesture classification (rock / paper / scissors)
- Game logic, scoring and user interface

Known limitations of the tracking, which affect what you will see:

- **Hand orientation.** All training recordings show the hand held side-on.
  An upright hand degrades tracking noticeably.
- **Lighting.** Training recordings were made in a well-lit room. Poor
  lighting degrades tracking noticeably.
- **Background.** Recordings were made against a dark background. Light
  backgrounds work but less reliably.
- **Left hand.** All training recordings use the right hand, so left hands
  are expected to track poorly. Not yet tested systematically.
- **Scissors with ring and little finger.** This variant is not
  represented in the recorded data and tracks very poorly. The ordinary
  index-and-middle scissors works normally.
- **Other gestures.** Only the three gestures used in the game are
  represented in the training data. Anything else is outside what the
  system is currently built for.

---

## 4. Troubleshooting

**Everything runs but very slowly, 100+ ms per frame**
You have the CPU build of PyTorch. Check with:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
A version ending in `+cpu` means you need to reinstall:
`pip uninstall -y torch torchvision`, then repeat the install command in
section 1. Note that `cu124` wheels are no longer built for current PyTorch
releases — that index URL silently gives you the CPU build. Use `cu128`.

**`ModuleNotFoundError: No module named 'mediapipe.solutions'`**
You have mediapipe 1.0.1 or newer. Run `pip install mediapipe==0.10.14`.

**Camera does not open, or opens a black window**
The scripts default to camera index 0. If you have more than one camera,
change `CAMERA_ID` at the top of the script. On Windows 11, also check
Settings → Privacy → Camera and allow access for desktop apps.

**PyCharm: imports resolve but running the script fails**
PyCharm keeps two interpreter settings. Set both
*Settings → Project → Python Interpreter* **and** the interpreter in the run
configuration to the `.venv` interpreter.

---

## Repository layout

```
src/
  detection/      hand detector, landmark model, crop geometry, runtime pipeline
  capture/        camera handling
  classification/ gesture classification (in progress)
  game/           game rules (in progress)
  ui/             display (in progress)
scripts/          entry points: data collection, preprocessing, evaluation
tests/            unit tests
models/           trained model weights
data/             datasets and recordings (not in git)
logs/             diagnostics output
```