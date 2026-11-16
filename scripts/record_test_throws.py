import json
import time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent

SESSION = "magnus_test2"
OUT = ROOT / "data" / "testset" / SESSION

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720
ROUND_SECONDS = 3.5

LABELS = {ord("r"): "rock", ord("p"): "paper", ord("s"): "scissors"}
NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

OUT.mkdir(parents=True, exist_ok=True)
next_round = len(list(OUT.glob("round_*")))

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

state = "idle"
frames = []
times = []
start = None
counts = {"rock": 0, "paper": 0, "scissors": 0}

print(f"Lagrer i {OUT}, starter på runde {next_round}")
print("mellomrom = ta opp, r/p/s = gest etter opptak, x = forkast, q = avslutt")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    t = time.perf_counter()

    if state == "rec":
        frames.append(frame.copy())
        times.append(t)
        if t - start > ROUND_SECONDS:
            state = "label"

    show = frame.copy()
    if state == "idle":
        text, color = "space = record", (255, 255, 255)
    elif state == "rec":
        text, color = f"RECORDING {ROUND_SECONDS - (t - start):.1f} s", (0, 0, 255)
    else:
        text, color = "r = rock   p = paper   s = scissors   x = discard", (0, 255, 255)
    cv2.putText(show, text, (20, 40), FONT, 1.0, color, 2)
    cv2.putText(show, f"round {next_round}   rock {counts['rock']}  paper {counts['paper']}  "
                f"scissors {counts['scissors']}", (20, show.shape[0] - 20),
                FONT, 0.7, (200, 200, 200), 2)
    cv2.imshow("Opptak testsett", show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if state == "idle" and key == ord(" "):
        frames, times, start = [], [], t
        state = "rec"

    elif state == "label" and key in LABELS:
        label = LABELS[key]
        folder = OUT / f"round_{next_round:03d}"
        folder.mkdir()
        for i, f in enumerate(frames):
            cv2.imwrite(str(folder / f"frame_{i:03d}.jpg"), f)
        with open(folder / "meta.json", "w") as fh:
            json.dump({"label": label, "times": times}, fh)
        counts[label] += 1
        fps = (len(times) - 1) / (times[-1] - times[0])
        print(f"Runde {next_round}: {NAMES[label]:6s} {len(frames)} bilder, {fps:.1f} fps")
        next_round += 1
        state = "idle"

    elif state == "label" and key == ord("x"):
        print("  forkastet")
        state = "idle"

cap.release()
cv2.destroyAllWindows()
print(f"Lagret denne gangen: stein {counts['rock']}, papir {counts['paper']}, "
      f"saks {counts['scissors']}")