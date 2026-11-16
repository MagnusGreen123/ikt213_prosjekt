
import csv
from collections import deque
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VIDEO = ROOT / "data" / "latency" / "IMG_0068.mov"
CAPTURE_FPS = 240
SHOW_W = 960
BUFFER = 2400
OUT = ROOT / "results" / "latency_hardware.csv"

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    print(f"Fant ikke eller kunne ikke åpne {VIDEO}")
    raise SystemExit
print(f"Videoen sier {cap.get(cv2.CAP_PROP_FPS):.1f} fps, vi regner med {CAPTURE_FPS} fps opptak")

frames = deque(maxlen=BUFFER)
pos = -1
playing = False
hand_frame = None
throws = []


def read_next():
    ok, img = cap.read()
    if not ok:
        return False
    number = frames[-1][0] + 1 if frames else 0
    h = int(img.shape[0] * SHOW_W / img.shape[1])
    frames.append((number, cv2.resize(img, (SHOW_W, h))))
    return True


def step(n):
    global pos
    pos += n
    while pos >= len(frames):
        if not read_next():
            pos = len(frames) - 1
            break
    pos = max(0, pos)


step(1)
while True:
    if playing:
        step(1)
    number, img = frames[pos]
    show = img.copy()
    t_ms = number / CAPTURE_FPS * 1000
    cv2.putText(show, f"bilde {number}   {t_ms:.0f} ms", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(show, f"bilde {number}   {t_ms:.0f} ms", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if hand_frame is not None:
        since = (number - hand_frame) / CAPTURE_FPS * 1000
        cv2.putText(show, f"ekte bevegelse i bilde {hand_frame}, +{since:.0f} ms", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(show, f"målinger: {len(throws)}", (15, show.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    cv2.imshow("Latens fra video", show)

    key = cv2.waitKey(30 if playing else 0) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        playing = not playing
    elif key == ord("d"):
        step(1)
    elif key == ord("a"):
        step(-1)
    elif key == ord("w"):
        step(24)
    elif key == ord("s"):
        step(-24)
    elif key == ord("h"):
        hand_frame = number
        playing = False
        print(f"Ekte bevegelse: bilde {number}")
    elif key == ord("r") and hand_frame is not None:
        ms = (number - hand_frame) / CAPTURE_FPS * 1000
        throws.append((hand_frame, number, ms))
        print(f"  Måling {len(throws)}: {ms:.0f} ms ({number - hand_frame} bilder)")
        hand_frame = None
    elif key == ord("u") and throws:
        print(f"  Angret måling {len(throws)}")
        throws.pop()

cap.release()
cv2.destroyAllWindows()

if throws:
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["real_motion_frame", "screen_motion_frame", "latency_ms"])
        writer.writerows(throws)
    ms = np.array([t[2] for t in throws])
    print(f"\nGlass-to-glass, {len(ms)} målinger: median {np.median(ms):.0f} ms, "
          f"min {ms.min():.0f} ms, maks {ms.max():.0f} ms")
    print(f"Ett bilde i videoen er {1000 / CAPTURE_FPS:.1f} ms, så hver måling er så nøyaktig")
    print(f"Lagret i {OUT}")