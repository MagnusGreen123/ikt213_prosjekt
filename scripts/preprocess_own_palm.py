import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.boxes import box_from_landmarks

REC_DIR = ROOT / "data" / "recordings"
OUT = ROOT / "data" / "processed" / "own_128_palm_rl.npz"

SIZE = 128

if OUT.exists():
    print(f"{OUT.name} finnes allerede, endre OUT eller slett fila først")
    raise SystemExit

OUT.parent.mkdir(parents=True, exist_ok=True)

images = []
keypoints = []
throw_ids = []
persons = []

throws = sorted(REC_DIR.glob("*.npz"))
print(f"Fant {len(throws)} kast")

for throw_nr, npz_path in enumerate(throws):
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    person = str(data["person"])
    img_dir = REC_DIR / npz_path.stem

    if not img_dir.exists():
        continue

    for i in range(len(landmarks)):
        img_path = img_dir / f"{i:04d}.jpg"
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        pts = landmarks[i, :, :2] * [w, h]

        x, y, side = box_from_landmarks(pts)
        left = int(x)
        top = int(y)
        side = int(side)
        if side <= 0:
            continue

        crop = np.zeros((side, side, 3), dtype=np.uint8)
        sx0, sy0 = max(0, left), max(0, top)
        sx1, sy1 = min(w, left + side), min(h, top + side)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        crop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = frame[sy0:sy1, sx0:sx1]

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (SIZE, SIZE))

        kp = (pts - [left, top]) / side

        images.append(crop)
        keypoints.append(kp.astype(np.float32))
        throw_ids.append(throw_nr)
        persons.append(person)

    if throw_nr % 10 == 0:
        print(f"  kast {throw_nr}/{len(throws)}, {len(images)} bilder så langt")

images = np.array(images, dtype=np.uint8)
keypoints = np.array(keypoints, dtype=np.float32)
throw_ids = np.array(throw_ids, dtype=np.int32)
persons = np.array(persons)

np.savez(OUT, images=images, keypoints=keypoints, throw_id=throw_ids, person=persons)

print(f"\nFerdig: {len(images)} bilder, {images.nbytes / 1e6:.0f} MB")
print(f"Personer: {dict(zip(*np.unique(persons, return_counts=True)))}")
print(f"Lagret til {OUT}")