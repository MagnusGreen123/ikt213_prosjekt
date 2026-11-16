

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline, draw_hand, draw_box

PERSON = "Alan"
RECORDINGS = ROOT / "data" / "ny"
DETECTOR_PATH = ROOT / "models" / "detector_palm.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy.pt"

PER_THROW = 6
ROWS = 8
TILE_W = 320
OUT = ROOT / "results"


def throw_folders():

    folders = set()
    for jpg in RECORDINGS.rglob("*.jpg"):
        if PERSON in str(jpg.parent.relative_to(RECORDINGS)):
            folders.add(jpg.parent)
    return sorted(folders)


folders = throw_folders()
if not folders:
    print(f"Fant ingen kast med '{PERSON}' i mappenavnet under {RECORDINGS}")
    sys.exit()
print(f"{len(folders)} kast for {PERSON}")

pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
OUT.mkdir(exist_ok=True)

rows = []
sheet = 0
found_all, total_all = 0, 0
for n, folder in enumerate(folders):
    files = sorted(folder.glob("*.jpg"))
    show = set(np.linspace(0, len(files) - 1, PER_THROW).astype(int))
    tiles = []
    found = 0
    for i, f in enumerate(files):
        frame = cv2.imread(str(f))
        pts, conf = pipe.process(frame)
        if pts is not None:
            found += 1
        if i in show:
            if pipe.last.get("box") is not None:
                draw_box(frame, pipe.last["box"], (0, 255, 0))
            if pts is not None:
                draw_hand(frame, pts)
            else:
                cv2.putText(frame, "NO HAND", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
            h = int(frame.shape[0] * TILE_W / frame.shape[1])
            tiles.append(cv2.resize(frame, (TILE_W, h)))
    found_all += found
    total_all += len(files)
    print(f"  {folder.name}: hånd funnet i {found}/{len(files)} bilder ({found / len(files) * 100:.0f} %)")

    while len(tiles) < PER_THROW:
        tiles.append(np.zeros_like(tiles[0]))
    row = np.hstack(tiles)
    cv2.putText(row, folder.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rows.append(row)
    if len(rows) == ROWS or n == len(folders) - 1:
        path = OUT / f"pipeline_{PERSON}_{DETECTOR_PATH.stem}_{sheet:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        print(f"Lagret {path}")
        rows = []
        sheet += 1

print(f"\nTotalt: hånd funnet i {found_all}/{total_all} bilder ({found_all / total_all * 100:.1f} %)")