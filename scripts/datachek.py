from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"

PERSON="magnus"


per_label = {}
per_person = {}
total_frames = 0
total_throws = 0

for npz_path in sorted(REC_DIR.glob("*.npz")):
    data = np.load(npz_path)
    label = str(data["label"])
    person = str(data["person"]) if "person" in data else "ukjent"
    if person != PERSON:
        continue

    img_dir = REC_DIR / npz_path.stem
    if img_dir.exists():
        n_frames = len(list(img_dir.glob("*.jpg")))
    else:
        n_frames = len(data["landmarks"])

    if label not in per_label:
        per_label[label] = {"kast": 0, "bilder": 0}
    per_label[label]["kast"] += 1
    per_label[label]["bilder"] += n_frames

    if person not in per_person:

        per_person[person] = {"kast": 0, "bilder": 0}
    per_person[person]["kast"] += 1
    per_person[person]["bilder"] += n_frames

    total_throws += 1
    total_frames += n_frames

print(f"{'gest':28s} {'kast':>6s} {'bilder':>8s}")
for label in sorted(per_label):
    s = per_label[label]
    print(f"{label:28s} {s['kast']:6d} {s['bilder']:8d}")
print(f"{'TOTALT':28s} {total_throws:6d} {total_frames:8d}")

print(f"\n{'person':28s} {'kast':>6s} {'bilder':>8s}")
for person in sorted(per_person):
    s = per_person[person]
    print(f"{person:28s} {s['kast']:6d} {s['bilder']:8d}")

print("\nLaTeX-rader til tab:dataset:")
for label in sorted(per_label):
    s = per_label[label]
    print(f"    {label:28s} & {s['kast']:6d} & {s['bilder']:8d} \\\\")
print(f"    {'Total':28s} & {total_throws:6d} & {total_frames:8d} \\\\")