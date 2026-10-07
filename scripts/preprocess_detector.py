from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"
OUT = ROOT / "data" / "processed" / "detector_192.npz"

SIZE = 192
MARGIN = 0.4

OUT.parent.mkdir(parents=True, exist_ok=True)

images = []
boxes = []
throw_ids = []

throws = sorted(REC_DIR.glob("*.npz"))
print(f"Fant {len(throws)} kast")

for throw_nr, npz_path in enumerate(throws):
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    img_dir = REC_DIR / npz_path.stem

    if not img_dir.exists():
        continue

    for i in range(len(landmarks)):
        frame = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        pts = landmarks[i, :, :2] * [w, h]

        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * (1 + MARGIN)

        small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(small, (SIZE, SIZE))

        images.append(small)
        boxes.append([cx / w, cy / h, side / w])
        throw_ids.append(throw_nr)

    if throw_nr % 10 == 0:
        print(f"  kast {throw_nr}/{len(throws)}, {len(images)} bilder")

images = np.array(images, dtype=np.uint8)
boxes = np.array(boxes, dtype=np.float32)
throw_ids = np.array(throw_ids, dtype=np.int32)

np.savez(OUT, images=images, boxes=boxes, throw_id=throw_ids)

print(f"\nFerdig: {len(images)} bilder, {images.nbytes / 1e6:.0f} MB")
print(f"Handstørrelse: min {boxes[:, 2].min():.3f}, median {np.median(boxes[:, 2]):.3f}, "
      f"maks {boxes[:, 2].max():.3f} (andel av bildebredden)")
print(f"Lagret til {OUT}")
