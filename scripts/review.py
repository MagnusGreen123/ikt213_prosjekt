

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.boxes import box_from_landmarks

REC_DIR = ROOT / "data" / "ny"
OUT_DIR = ROOT / "data" / "screening"
REJECT_DIR = OUT_DIR / "avvist"

PERSON = "alan"
TILES = 8
TILE_W, TILE_H = 320, 180

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(exist_ok=True)


def draw(img, pts, box):
    x, y, side = box
    inside = ((pts[:, 0] >= x) & (pts[:, 0] <= x + side) &
              (pts[:, 1] >= y) & (pts[:, 1] <= y + side)).all()

    color = (0, 255, 0) if inside else (0, 0, 255)
    cv2.rectangle(img, (int(x), int(y)), (int(x + side), int(y + side)), color, 6)

    for a, b in CONNECTIONS:
        cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                 (int(pts[b, 0]), int(pts[b, 1])), (255, 255, 0), 5)
    for px, py in pts:
        cv2.circle(img, (int(px), int(py)), 8, (0, 255, 255), -1)

    return inside



rejected = {p.stem for p in REJECT_DIR.glob("*.jpg")}

paths = sorted(REC_DIR.glob(f"*_{PERSON}_*.npz"))
print(f"{len(paths)} kast fra {PERSON}, {len(rejected)} allerede avvist\n")

for path in paths:
    name = path.stem
    if name in rejected:
        continue

    landmarks = np.load(path)["landmarks"]
    n = len(landmarks)

    tiles = []
    outside = 0
    for i in np.linspace(0, n - 1, TILES).astype(int):
        img = cv2.imread(str(REC_DIR / name / f"{i:04d}.jpg"))
        if img is None:
            img = np.zeros((720, 1280, 3), np.uint8)

        h, w = img.shape[:2]
        pts = landmarks[i, :, :2] * np.array([w, h])

        if not draw(img, pts, box_from_landmarks(pts)):
            outside += 1

        cv2.putText(img, str(i), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        tiles.append(cv2.resize(img, (TILE_W, TILE_H)))

    half = TILES // 2
    sheet = np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])])

    header = np.zeros((40, sheet.shape[1], 3), np.uint8)
    text = f"{name}   {n} frames   box too small: {outside} of {TILES}"
    cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(str(OUT_DIR / f"{name}.jpg"), np.vstack([header, sheet]))
    print(f"  {name:28s} boks for liten på {outside} av {TILES} bilder")

print(f"\nArkene ligger i {OUT_DIR}")
print(f"Flytt kastene som skal ut til {REJECT_DIR}")