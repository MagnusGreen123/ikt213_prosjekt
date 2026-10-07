import random
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"

SAMPLES = 30
SECONDS = 2.0
ONLY_LATE = False

BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def throw_number(name):
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


throws = sorted(REC_DIR.glob("*.npz"))
if ONLY_LATE:
    throws = [t for t in throws if throw_number(t.stem) >= 80]

if not throws:
    raise SystemExit(f"Fant ingen .npz-filer i {REC_DIR}")

print(f"Fant {len(throws)} kast")

candidates = []
for npz_path in throws:
    img_dir = REC_DIR / npz_path.stem
    if not img_dir.exists():
        continue
    n_images = len(list(img_dir.glob("*.jpg")))
    for i in range(n_images):
        candidates.append((npz_path, i))

if not candidates:
    raise SystemExit("Fant ingen bilder. Ble SAVE_IMAGES satt til False?")

print(f"Fant {len(candidates)} bilder totalt")

picks = random.sample(candidates, min(SAMPLES, len(candidates)))

print(f"\nViser {len(picks)} tilfeldige bilder, {SECONDS} s hver")
print("mellomrom = neste med en gang, q = avslutt\n")

for nr, (npz_path, index) in enumerate(picks, start=1):
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    times = data["times"]
    label = str(data["label"])

    if index >= len(landmarks):
        continue

    img_path = REC_DIR / npz_path.stem / f"{index:04d}.jpg"
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, z in landmarks[index]]

    for i, (a, b) in enumerate(BONES):
        cv2.line(frame, points[a], points[b], COLORS[i // 4], 2)
    for p in points:
        cv2.circle(frame, p, 5, (255, 255, 255), -1)
        cv2.circle(frame, p, 5, (0, 0, 0), 1)

    andel = times[index] / times[-1] if times[-1] > 0 else 0
    tekst = [
        f"{nr}/{len(picks)}  {npz_path.stem}",
        f"bilde {index}/{len(landmarks) - 1}  t={times[index]:.2f}s ({andel*100:.0f}% av kastet)",
        f"label: {label}",
    ]
    y = 30
    for line in tekst:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30

    print(f"{nr}/{len(picks)}  {npz_path.stem}  bilde {index}  ({andel*100:.0f}% av kastet)")

    cv2.imshow("Sjekk merking", frame)
    key = cv2.waitKey(int(SECONDS * 1000)) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("\nFerdig")
