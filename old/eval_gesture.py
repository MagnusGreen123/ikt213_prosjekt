from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"

THRESHOLD = 0.5
END_FRAMES = 5
GESTURES = ["rock", "paper", "scissors"]
FINGER_BASE = [1, 5, 9, 13, 17]
COUNT_TO_GESTURE = {0: "rock", 4: "paper", 2: "scissors"}


def finger_extension(pts):
    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-9
    return np.array([np.linalg.norm(pts[b + 3] - pts[b]) / scale for b in FINGER_BASE])


def classify(ext):
    extended = int((ext[1:] > THRESHOLD).sum())
    return COUNT_TO_GESTURE.get(extended, "unsure")


def image_size(name):
    img = cv2.imread(str(REC_DIR / name / "0000.jpg"))
    if img is None:
        return 1280, 720
    return img.shape[1], img.shape[0]


results = {}

for path in sorted(REC_DIR.glob("*.npz")):
    data = np.load(path)
    label = str(data["label"])
    person = str(data["person"])
    times = data["times"]
    w, h = image_size(path.stem)
    pts = data["landmarks"][:, :, :2] * np.array([w, h])

    per_frame = [classify(finger_extension(p)) for p in pts]

    end_ext = np.median([finger_extension(p) for p in pts[-END_FRAMES:]], axis=0)
    answer = classify(end_ext)

    stable_from = len(per_frame)
    while stable_from > 0 and per_frame[stable_from - 1] == label:
        stable_from -= 1
    if stable_from < len(per_frame):
        stable_ms = (times[-1] - times[stable_from]) * 1000
    else:
        stable_ms = 0.0

    results.setdefault(person, []).append((label, answer, stable_ms))


for person, rows in results.items():
    labels = np.array([r[0] for r in rows])
    answers = np.array([r[1] for r in rows])
    stable = np.array([r[2] for r in rows])

    print(f"\n=== {person}: {len(rows)} kast, "
          f"riktig i sluttposen {np.mean(labels == answers) * 100:.1f} % ===")

    print(f"  {'fasit':10s} " + " ".join(f"{g:>9s}" for g in GESTURES + ['unsure']))
    for g in GESTURES:
        row = [np.sum((labels == g) & (answers == a)) for a in GESTURES + ["unsure"]]
        print(f"  {g:10s} " + " ".join(f"{n:9d}" for n in row))

    for g in GESTURES:
        ok = (labels == g) & (answers == g)
        if ok.any():
            print(f"  {g:10s} riktig og stabilt i median {np.median(stable[ok]):.0f} ms "
                  f"før slutten av kastet")