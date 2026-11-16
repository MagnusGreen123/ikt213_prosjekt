from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
NEG_DIR = ROOT / "data" / "negatives"
OUT = ROOT / "data" / "processed" / "negatives_192.npz"

SIZE = 192
GROUP = 100  

images = []
groups = []
group_nr = 0

sessions = sorted(p for p in NEG_DIR.iterdir() if p.is_dir())
print(f"Fant {len(sessions)} opptak")

for session in sessions:
    files = sorted(session.glob("*.jpg"))
    for i, path in enumerate(files):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(small, (SIZE, SIZE))

        images.append(small)
        groups.append(group_nr + i // GROUP)

    group_nr += len(files) // GROUP + 1
    print(f"  {session.name}: {len(files)} bilder")

images = np.array(images, dtype=np.uint8)
groups = np.array(groups, dtype=np.int32)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, images=images, group=groups)

print(f"\nFerdig: {len(images)} bilder i {len(np.unique(groups))} grupper, "
      f"{images.nbytes / 1e6:.0f} MB")
print(f"Lagret til {OUT}")