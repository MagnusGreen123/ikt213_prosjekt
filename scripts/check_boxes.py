import sys
import random
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.boxes import box_from_landmarks

REC_DIR = ROOT / "data" / "recordings"
PERSON = "Christian"
SAMPLES = 48
COLS = 8
TILE_W = 320
OUT = ROOT / "results" / f"check_boxes_{PERSON}.jpg"

random.seed(0)
candidates = []
for npz_path in sorted(REC_DIR.glob("*.npz")):
    data = np.load(npz_path)
    if str(data["person"]) != PERSON:
        continue
    img_dir = REC_DIR / npz_path.stem
    for i in range(len(data["landmarks"])):
        if (img_dir / f"{i:04d}.jpg").exists():
            candidates.append((npz_path, i))
print(f"{len(candidates)} bilder med punkter for {PERSON}")
if not candidates:
    raise SystemExit

tiles = []
for npz_path, i in random.sample(candidates, min(SAMPLES, len(candidates))):
    frame = cv2.imread(str(REC_DIR / npz_path.stem / f"{i:04d}.jpg"))
    h, w = frame.shape[:2]
    pts = np.load(npz_path)["landmarks"][i, :, :2] * [w, h]
    left, top, side = box_from_landmarks(pts)
    cv2.rectangle(frame, (int(left), int(top)), (int(left + side), int(top + side)), (0, 255, 0), 4)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"{npz_path.stem} #{i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    th = int(h * TILE_W / w)
    tiles.append(cv2.resize(frame, (TILE_W, th)))

while len(tiles) % COLS:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]
OUT.parent.mkdir(exist_ok=True)
cv2.imwrite(str(OUT), np.vstack(rows))
print(f"Lagret {OUT}")