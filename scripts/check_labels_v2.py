
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings" / "recordings"

PERSON = "christian"
WIDTH = 1280
HEIGHT = 720
THRESHOLD = 0.5
END_FRAMES = 5

EXPECTED = {"rock": 0, "paper": 4, "scissors": 2}
FINGER_BASE = [1, 5, 9, 13, 17]


def finger_extension(pts):
    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-9
    return np.array([np.linalg.norm(pts[b + 3] - pts[b]) / scale for b in FINGER_BASE])


wrong = []
paths = sorted(REC_DIR.glob(f"*_{PERSON}_*.npz"))

for path in paths:
    data = np.load(path)
    label = str(data["label"])

    pts = data["landmarks"][:, :, :2] * np.array([WIDTH, HEIGHT])

    end = pts[-END_FRAMES:]
    ext = np.median([finger_extension(p) for p in end], axis=0)
    extended = int((ext[1:] > THRESHOLD).sum())

    status = "ok" if extended == EXPECTED[label] else "FEIL"
    print(f"{path.stem:28s} {label:9s} utstrakte {extended}  "
          f"{np.round(ext[1:], 2)}  {status}")
    if status == "FEIL":
        wrong.append(path.stem)

print(f"\n{len(wrong)} av {len(paths)} kast har en sluttpose som ikke stemmer med etiketten")
for name in wrong:
    print("  ", name)