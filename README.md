# Rock-Paper-Scissors Gesture Recognition

IKT213 Machine Vision, Group 13

A webcam game of rock, paper, scissors against the computer. The system tracks
21 joints of the hand in real time, counts the extended fingers, and follows
the pumping motion to decide which gesture is thrown, in most cases before the
hand has stopped.

The hand tracking is our own: a ResNet-18 hand detector and a heatmap landmark
model that we trained ourselves, following the same two-stage architecture as
MediaPipe Hands. MediaPipe is used as a labelling tool and as a reference for
comparison, not as a part of the running system.

The full description, method and results are in the final report.

---

## 1. Installation

Requirements:

- Windows 11
- Python 3.12
- A webcam
- NVIDIA GPU with CUDA 12.x. Optional, but gives about 14 ms per frame. On
  the CPU the game runs at about 25 frames per second.

Clone and create a virtual environment:

```bash
git clone https://github.com/MagnusGreen123/ikt213_prosjekt.git
cd ikt213_prosjekt
python -m venv .venv
.venv\Scripts\activate
```

**Install PyTorch first.** It is not in `requirements.txt`, because it has to
come from its own package index to get GPU support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Without an NVIDIA GPU, leave out the `--index-url` part. Tested with
torch 2.11.0+cu128 and torchvision 0.26.0+cu128.

Then the rest:

```bash
pip install -r requirements.txt
```

The trained model weights are in `models/`, so there is nothing to download.
The game and the evaluation scripts use:

| File | Model |
|---|---|
| `detector_palm_neg_v2.pt` | Hand detector, final version (v2) |
| `finetuned_palm_frei_v4_aug_heavy_magrl.pt` | Landmark model, final version |

---

## 2. Playing the game

```bash
python scripts/play_game.py
```

Pump your fist twice and throw on the third down-stroke. The game locks your
gesture, shows the computer's move and keeps score. The first to win two
rounds wins the match.

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Restart the round |
| `n` | New match |
| `c` | Switch between fair and unbeatable mode |
| `f` | Toggle full screen |
| `s` | Toggle sound |

In the **fair mode** the computer picks its move before you throw. In the
**unbeatable mode** it waits for your locked gesture and answers with the move
that beats it. The mode is set with `UNBEATABLE` at the top of the script.

Both scissors variants work: index and middle finger, or ring and little
finger. Both hands work.

**For the best result:** good light, a wall fairly close behind the hand, the
hand roughly 50 cm from the camera, and only the throwing hand in view. A white
wall in dim light does not work: every throw is read as paper.

For each lock, the game saves the last frames with landmarks and finger values
to `results/locks/`. Turn this off with `SAVE_LOCKS = False`.

---

## 3. Measuring and recording

### Live test

```bash
python scripts/eval/live_gesture.py
```

Measures accuracy and latency live, using the same decider as the game. Keys:
`f` marks the last round as wrong, `r` resets, `2`/`3` sets the number of
frames needed to lock, `q` quits and prints the summary.

### Recording data

| Script | What it does |
|---|---|
| `scripts/collect_data.py` | Records labelled throws with MediaPipe landmarks to `data/recordings/`. Set `PERSON`, `MODE` and `THROWS` at the top. Space starts, `u` undoes the last throw, `q` quits. |
| `scripts/record_test_throws.py` | Records complete rounds from the first pump until after the hand has stopped, to `data/testset/`. Used to tune the throw decider. |
| `scripts/record_neg.py` | Records frames without a hand, used as negative examples for the detector. |

### Evaluation scripts (`scripts/eval/`)

| Script | What it does |
|---|---|
| `eval_gest_model.py` | Gesture accuracy per throw on recorded throws, for our models and MediaPipe |
| `eval_models.py` | Landmark error of all landmark models on our data and on FreiHAND |
| `sweep_decider.py` | Runs the throw decider on the recorded rounds for a grid of parameters |
| `debug_round.py` | Prints the decider's state frame by frame for one round |
| `lock_vs_form.py` | When the gesture is locked relative to when the fingers have finished forming |
| `early_analysis.py` | Finger values over time around the moment the hand stops |
| `report_numbers.py` | Recomputes the offline numbers in the report to `results/report/` |
| `report_figures.py` | Makes the data-based figures in the report |
| `measure_latency_video.py` | Glass-to-glass latency from a 240 fps phone video |
| `diagnose_full_v2.py` | The First Release diagnostics against MediaPipe |

The scripts that use recorded data need `data/`, which is not in the
repository (see the end of this file).

---

## 4. Troubleshooting

**Everything runs but very slowly, 100+ ms per frame**
You have the CPU build of PyTorch. Check with:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
A version ending in `+cpu` means you need to reinstall:
`pip uninstall -y torch torchvision`, then repeat the install command in
section 1. The `cu124` index no longer has builds for current PyTorch
releases and silently gives you the CPU build. Use `cu128`.

**`ModuleNotFoundError: No module named 'mediapipe.solutions'`**
You have a newer MediaPipe. Run `pip install mediapipe==0.10.14`.

**Camera does not open, or opens a black window**
The scripts use camera index 0. If you have more than one camera, change
`CAMERA_ID` at the top of the script. On Windows 11, also check
Settings → Privacy → Camera and allow access for desktop apps.

**`FileNotFoundError` for a model file**
Check that `models/` contains the two files in section 1.

**PyCharm: imports resolve but running the script fails**
Set both *Settings → Project → Python Interpreter* and the interpreter in the
run configuration to the `.venv` interpreter.

---

## Repository layout

```
src/
  detection/       detector, landmark model, crop geometry (boxes.py),
                   runtime pipeline and training code
  classification/  finger rule and throw decider (gesture.py),
                   lock snapshots, test set loading
scripts/           the game, data recording and preprocessing
  eval/            evaluation and measurement scripts
old/               scripts no longer in use, kept for reference
models/            trained model weights
data/              recordings, test rounds and FreiHAND (not in git)
results/           output from the game and the evaluation scripts
```

`data/` is not in the repository because of its size and because it contains
recordings of people. FreiHAND can be downloaded from its authors.
