import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.classification import gesture_2 as G
from src.classification.testset import load_session, reference_rest

DATA = ROOT / "data" / "testset"
SESSION = "magnus_test1"
ROUNDS = [2, 0]
DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

rounds = load_session(DATA / SESSION, DETECTOR_PATH, LANDMARK_PATH)
print("Regel:", "retthet" if G.USE_STRAIGHTNESS else "fingerstrekk",
      " MARGIN", G.MARGIN, " FRAMES", G.FRAMES, " EARLY", G.EARLY)

for number in ROUNDS:
    r = rounds[number]
    rest = reference_rest(r)
    if rest is None:
        rest_text = "ingen ro funnet, tid fra start"
    else:
        rest_text = f"ro ved {(rest - r['times'][0]) * 1000:.0f} ms fra start"
    print(f"\n=== {r['name']}  fasit {r['label']}  {rest_text} ===")
    print("   ms  pumper retning vindu trege I    M    R    L    gest      sikk  stille rekke  i ro  over bunn")
    d = G.ThrowDecider()
    shown = False
    for k, (p, t) in enumerate(zip(r["pts"], r["times"])):
        decision = d.update(p, t)
        ms = (t - rest) * 1000 if rest is not None else (t - r["times"][0]) * 1000
        if d.pumps < 1 and not shown:
            continue
        shown = True
        if p is None:
            print(f"{ms:6.0f}  ingen hånd")
            continue
        values, threshold = G.finger_values(p)
        g, c = G.classify_frame(p)
        above = "" if d.bottom_y is None else f"{d.bottom_y - d.prev_y:5.2f}"
        vals = " ".join(f"{v:.2f}" for v in values)
        print(f"{ms:6.0f}  {d.pumps:5d}  {str(d.direction):6s}  {'ja ' if d.window_open else 'nei'}   "
              f"{d.slow_frames:3d}   {vals}  {str(g):9s} {c:.2f}  {'ja ' if d.last_still else 'nei'}    "
              f"{d.streak:3d}    {'ja' if d.slow_frames >= G.REST_FRAMES else '  '}  {above}")
        if decision is not None:
            print(f"        LÅST {decision}")
            break