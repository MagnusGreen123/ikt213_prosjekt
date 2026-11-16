

import json
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "ny"
REJECT_DIR = ROOT / "data" / "recordings_avvist"
DECISIONS = ROOT / "data" / "review_decisions.json"

PERSON = "alan"
FRAMES = 10
COLS = 5
TILE = 300
MARGIN = 1.8
REVIEW_ALL = True

FINGERS = [
    ([1, 2, 3, 4], (0, 0, 255)),
    ([5, 6, 7, 8], (0, 165, 255)),
    ([9, 10, 11, 12], (0, 255, 255)),
    ([13, 14, 15, 16], (0, 255, 0)),
    ([17, 18, 19, 20], (255, 0, 0)),
]
PALM = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

WINDOW = "Gjennomgang"


def pt(p):
    return int(p[0]), int(p[1])


def make_tile(img, pts):
    h, w = img.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(max(x1 - x0, y1 - y0, 100) * MARGIN)
    left = int(cx - side / 2)
    top = int(cy - side / 2)


    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = img[sy0:sy1, sx0:sx1]

    tile = cv2.resize(canvas, (TILE, TILE))
    p = (pts - np.array([left, top])) * (TILE / side)

    for a, b in PALM:
        cv2.line(tile, pt(p[a]), pt(p[b]), (255, 255, 255), 2)
    for chain, color in FINGERS:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(tile, pt(p[a]), pt(p[b]), color, 2)
        for i in chain:
            cv2.circle(tile, pt(p[i]), 3, color, -1)
    cv2.circle(tile, pt(p[0]), 4, (255, 255, 255), -1)

    return tile


def make_sheet(name, label, landmarks, number, total, previous):
    n = len(landmarks)


    picks = np.unique(np.linspace(0, n - 1, FRAMES).round().astype(int))

    tiles = []
    for i in picks:
        img = cv2.imread(str(REC_DIR / name / f"{i:04d}.jpg"))
        if img is None:
            img = np.zeros((720, 1280, 3), np.uint8)
        h, w = img.shape[:2]
        pts = landmarks[i, :, :2] * np.array([w, h])

        tile = make_tile(img, pts)
        cv2.putText(tile, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)

    while len(tiles) < FRAMES:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))

    rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, FRAMES, COLS)]
    sheet = np.vstack(rows)

    header = np.zeros((70, sheet.shape[1], 3), np.uint8)
    title = f"{number}/{total}   {name}   ({label}, {n} frames)"
    if previous:
        title += f"   [previous: {previous}]"
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(header, "g = good   d = bad   b = back   q = quit",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    return np.vstack([header, sheet])


decisions = {}
if DECISIONS.exists():
    decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))

names = [p.stem for p in sorted(REC_DIR.glob(f"*_{PERSON}_*.npz"))]
print(f"{len(names)} kast fra {PERSON}, {sum(n in decisions for n in names)} allerede vurdert")


pos = 0
if not REVIEW_ALL:
    while pos < len(names) and names[pos] in decisions:
        pos += 1

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

while 0 <= pos < len(names):
    name = names[pos]
    data = np.load(REC_DIR / f"{name}.npz")
    sheet = make_sheet(name, str(data["label"]), data["landmarks"],
                       pos + 1, len(names), decisions.get(name))
    cv2.imshow(WINDOW, sheet)

    key = cv2.waitKey(0) & 0xFF


    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord("g"):
        decisions[name] = "keep"
        pos += 1
    elif key == ord("d"):
        decisions[name] = "reject"
        pos += 1
    elif key == ord("b"):
        pos = max(0, pos - 1)
        continue
    elif key == ord("q"):
        break
    else:
        continue

    DECISIONS.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

cv2.destroyAllWindows()


REJECT_DIR.mkdir(parents=True, exist_ok=True)
moved = 0
for name, verdict in decisions.items():
    if verdict != "reject":
        continue
    npz = REC_DIR / f"{name}.npz"
    if not npz.exists():
        continue
    shutil.move(str(npz), str(REJECT_DIR / npz.name))
    img_dir = REC_DIR / name
    if img_dir.exists():
        shutil.move(str(img_dir), str(REJECT_DIR / name))
    moved += 1

reviewed = [n for n in names if n in decisions]
kept = [n for n in reviewed if decisions[n] == "keep"]

print(f"\nVurdert: {len(reviewed)} av {len(names)}")
print(f"Beholdt: {len(kept)}")
print(f"Flyttet til {REJECT_DIR.name}: {moved}")
for gesture in ["paper", "rock", "scissors"]:
    print(f"  {gesture}: {sum(n.startswith(gesture) for n in kept)} beholdt")