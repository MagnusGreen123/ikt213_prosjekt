from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"

MARGIN = 0.4

ratios = []
per_label = {}

for npz_path in sorted(REC_DIR.glob("*.npz")):
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    label = str(data["label"])

    for pts in landmarks:
        p = pts[:, :2]

        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        side = max(x1 - x0, y1 - y0) * (1 + MARGIN)

        palm = np.linalg.norm(p[9] - p[0])
        if palm < 1e-6:
            continue

        ratios.append(side / palm)
        per_label.setdefault(label, []).append(side / palm)

ratios = np.array(ratios)

print(f"{len(ratios)} bilder\n")
print("Forhold mellom bokssiden og handflatens størrelse:")
print(f"  median: {np.median(ratios):.2f}")
print(f"  snitt:  {ratios.mean():.2f}")
print(f"  5. til 95. persentil: {np.percentile(ratios, 5):.2f} til {np.percentile(ratios, 95):.2f}")

print("\nPer gest:")
for label, vals in sorted(per_label.items()):
    vals = np.array(vals)
    print(f"  {label:10s} median {np.median(vals):.2f}, "
          f"5-95 %: {np.percentile(vals, 5):.2f} til {np.percentile(vals, 95):.2f}")

print(f"\nForeslått PALM_TO_BOX = {np.percentile(ratios, 75):.2f}")
print("(75. persentil, så boksen heller blir litt for stor enn for liten)")
