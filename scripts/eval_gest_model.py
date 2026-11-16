import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline
from src.classification import gesture as G

REC_DIR = ROOT / "data" / "recordings"
DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

PERSONS = ["magnus2"]

END_FRAMES = 5
GESTURES = ["rock", "paper", "scissors"]
FINGER_BASE = [1, 5, 9, 13, 17]
COUNT_TO_GESTURE = {0: "rock", 4: "paper", 2: "scissors"}


def finger_values(pts):
    values, threshold = G.finger_values(pts)
    return values


THRESHOLD = G.STRAIGHT_THRESHOLD if G.USE_STRAIGHTNESS else G.THRESHOLD


def classify(values):
    extended = int((values > THRESHOLD).sum())
    return COUNT_TO_GESTURE.get(extended, "unsure")


pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
print("Enhet:", pipe.device)
print("Regel:", "retthet" if G.USE_STRAIGHTNESS else "fingerstrekk", "terskel", THRESHOLD)
print("Detektor:", DETECTOR_PATH.name, "  landmarks:", LANDMARK_PATH.name)

results = {}

for path in sorted(REC_DIR.glob("*.npz")):
    data = np.load(path)
    person = str(data["person"])
    if person not in PERSONS:
        continue

    label = str(data["label"])
    mp_landmarks = data["landmarks"]
    img_dir = REC_DIR / path.stem

    pipe.reset()
    ours = []
    mp_pixels = []
    for i in range(len(mp_landmarks)):
        frame = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        pts, conf = pipe.process(frame)
        ours.append(pts)
        mp_pixels.append(mp_landmarks[i, :, :2] * np.array([w, h]))

    found = [p for p in ours if p is not None]
    missing = 1 - len(found) / max(len(ours), 1)

    if len(found) >= END_FRAMES:
        end_ext = np.median([finger_values(p) for p in found[-END_FRAMES:]], axis=0)
        answer = classify(end_ext)
    else:
        answer = "no hand"

    mp_end = np.median([finger_values(p) for p in mp_pixels[-END_FRAMES:]], axis=0)
    mp_answer = classify(mp_end)

    results.setdefault(person, []).append((label, answer, mp_answer, missing))
    if answer != label:
        print(f"  {path.stem:28s} fasit {label:9s} vi {answer:9s} MediaPipe {mp_answer}")


answers_all = GESTURES + ["unsure", "no hand"]
for person, rows in results.items():
    labels = np.array([r[0] for r in rows])
    ours = np.array([r[1] for r in rows])
    mp = np.array([r[2] for r in rows])
    missing = np.array([r[3] for r in rows])

    print(f"\n=== {person}: {len(rows)} kast ===")
    print(f"  riktig med våre modeller: {np.mean(labels == ours) * 100:.1f} %")
    print(f"  riktig med MediaPipe:     {np.mean(labels == mp) * 100:.1f} %")
    print(f"  bilder uten funnet hånd:  {np.mean(missing) * 100:.1f} %")

    print(f"  {'fasit':10s} " + " ".join(f"{g:>9s}" for g in answers_all))
    for g in GESTURES:
        row = [np.sum((labels == g) & (ours == a)) for a in answers_all]
        print(f"  {g:10s} " + " ".join(f"{n:9d}" for n in row))