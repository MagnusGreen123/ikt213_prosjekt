

import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"

SOURCE = Path(r"C:\Users\magnu\ctf\christian_kast")
NEW_PERSON = "christian2"

if not SOURCE.exists():
    print(f"Fant ikke {SOURCE}")
    raise SystemExit

copied = 0
skipped = 0
for npz in sorted(SOURCE.glob("*.npz")):
    img_src = SOURCE / npz.stem
    if not img_src.exists():
        print(f"  {npz.stem}: mangler bildemappe, hopper over")
        skipped += 1
        continue

    data = dict(np.load(npz))
    label = str(data["label"])
    number = npz.stem.split("_")[-1]
    new_name = f"{label}_{NEW_PERSON}_{number}"

    if (REC_DIR / f"{new_name}.npz").exists() or (REC_DIR / new_name).exists():
        print(f"  {new_name} finnes allerede, hopper over")
        skipped += 1
        continue

    data["person"] = np.array(NEW_PERSON)
    np.savez(REC_DIR / f"{new_name}.npz", **data)
    shutil.copytree(img_src, REC_DIR / new_name)
    copied += 1

print(f"\nKopierte {copied} kast som '{NEW_PERSON}', hoppet over {skipped}")