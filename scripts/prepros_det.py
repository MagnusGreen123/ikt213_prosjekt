import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.boxes import box_from_landmarks

REC_DIR = ROOT / "data" / "recordings"
OUT = ROOT / "data" / "processed" / "detector_192_palm_v2.npz"

SIZE = 192

if OUT.exists():
    print(f"{OUT.name} finnes allerede, endre OUT eller slett fila først")
    raise SystemExit

OUT.parent.mkdir(parents=True, exist_ok=True)

images = []
boxes = []
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
        frame = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        pts = landmarks[i, :, :2] * [w, h]

        left, top, side = box_from_landmarks(pts)
        cx = left + side / 2
        cy = top + side / 2

        small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(small, (SIZE, SIZE))

        images.append(small)
        boxes.append([cx / w, cy / h, side / w])
        throw_ids.append(throw_nr)
        persons.append(person)

    if throw_nr % 10 == 0:
        print(f"  kast {throw_nr}/{len(throws)}, {len(images)} bilder")

images = np.array(images, dtype=np.uint8)
boxes = np.array(boxes, dtype=np.float32)
throw_ids = np.array(throw_ids, dtype=np.int32)
persons = np.array(persons)

np.savez(OUT, images=images, boxes=boxes, throw_id=throw_ids, person=persons)

print(f"\nFerdig: {len(images)} bilder, {images.nbytes / 1e6:.0f} MB")
print(f"Personer: {dict(zip(*np.unique(persons, return_counts=True)))}")
print(f"Lagret til {OUT}")