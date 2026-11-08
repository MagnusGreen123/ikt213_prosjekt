
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline, draw_hand, draw_box
from src.classification import gesture as G

DETECTOR_PATH = ROOT / "models" / "detector_palm_neg.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy.pt"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720
SHOW_RESULT = 2.0
LIMIT_MS = 100

NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class RestTimer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_y = None
        self.prev_t = None
        self.slow = 0
        self.slow_start = None
        self.rest_time = None

    def update(self, pts, t, palm):
        if self.rest_time is not None or pts is None or palm is None:
            return
        y = pts[G.PALM_POINTS, 1].mean() / palm
        if self.prev_y is not None and t > self.prev_t:
            speed = abs(y - self.prev_y) / (t - self.prev_t)
            if speed < G.REST_SPEED:
                if self.slow == 0:
                    self.slow_start = t
                self.slow += 1
                if self.slow >= G.REST_FRAMES:
                    self.rest_time = self.slow_start
            else:
                self.slow = 0
        self.prev_y, self.prev_t = y, t


pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
decider = G.ThrowDecider()
rest = RestTimer()
rest_window = None
print("Enhet:", pipe.device)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


rounds = []
result_time = None
waiting_for_rest = False

print(f"Pump {decider.pumps_needed} ganger og kast. f = feil, r = ny runde, "
      f"2/3 = bilder på rad, q = avslutt")


def print_latency(r):
    lat = r["latency"]
    if lat is None:
        print(f"  runde {len(rounds)}: fant ikke ro før neste runde")
    else:
        when = "før" if lat < 0 else "etter"
        print(f"  runde {len(rounds)}: resultat {abs(lat):.0f} ms {when} at hånden stoppet")


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    t = time.perf_counter()

    pts, conf = pipe.process(frame)
    decision = decider.update(pts, t)

    if decider.window_time != rest_window:
        rest_window = decider.window_time
        rest.reset()
    if decider.window_time is not None:
        rest.update(pts, t, decider.palm)

    if decision is not None and result_time is None:
        result_time = t
        window_ms = (decider.decision_time - decider.window_time) * 1000
        rounds.append({"gesture": decision, "window_ms": window_ms, "right": True,
                       "lock_t": decider.decision_time,
                       "proc_ms": pipe.last.get("total_ms", 0), "latency": None})
        waiting_for_rest = True
        print(f"Runde {len(rounds)}: {NAMES[decision]:6s} {decider.pumps} pumper, "
              f"låst {window_ms:.0f} ms etter at vinduet åpnet")

    if waiting_for_rest and rest.rest_time is not None:
        r = rounds[-1]
        r["latency"] = (r["lock_t"] - rest.rest_time) * 1000 + r["proc_ms"]
        waiting_for_rest = False
        print_latency(r)

    if result_time is not None and t - result_time > SHOW_RESULT:
        if waiting_for_rest:
            waiting_for_rest = False
            print_latency(rounds[-1])
        decider.reset()
        result_time = None

    if pipe.last.get("box") is not None:
        draw_box(frame, pipe.last["box"], (0, 255, 0))
    if pts is not None:
        draw_hand(frame, pts)
        g, c = G.classify_frame(pts)
        cv2.putText(frame, f"frame: {g or '-'}   confidence {c:.2f}", (20, 80),
                    FONT, 0.8, (255, 255, 255), 2)

    if decider.decision is not None:
        text, color = NAMES[decider.decision], (0, 0, 255)
        size = cv2.getTextSize(text, FONT, 4.0, 8)[0]
        cv2.putText(frame, text, ((frame.shape[1] - size[0]) // 2, frame.shape[0] // 2),
                    FONT, 4.0, color, 8)
    elif decider.window_open:
        streak = f"{decider.streak_gesture or '-'} x{decider.streak}"
        cv2.putText(frame, f"WINDOW OPEN - throw!   {streak}", (20, 40),
                    FONT, 1.0, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"pumps: {decider.pumps}/{decider.pumps_needed}", (20, 40),
                    FONT, 1.0, (255, 255, 255), 2)

    h = frame.shape[0]
    cv2.putText(frame, f"frames to lock: {decider.frames_needed}   "
                f"{pipe.last.get('total_ms', 0):.0f} ms/frame", (20, h - 50),
                FONT, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, "f = wrong   r = reset   2/3 = frames   q = quit", (20, h - 20),
                FONT, 0.7, (200, 200, 200), 2)
    cv2.imshow("Stein saks papir", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        decider.reset()
        result_time = None
        waiting_for_rest = False
    elif key == ord("f") and rounds:
        rounds[-1]["right"] = False
        print(f"  runde {len(rounds)} merket som feil")
    elif key in (ord("2"), ord("3")):
        decider.frames_needed = int(chr(key))
        print(f"Låser nå på {decider.frames_needed} bilder på rad")

cap.release()
cv2.destroyAllWindows()

if rounds:
    right = sum(r["right"] for r in rounds)
    print(f"\n{len(rounds)} runder, {right} riktige ({right / len(rounds) * 100:.0f} %)")
    print(f"Latens = låst - ro + behandlingstid. Krav: {LIMIT_MS} ms")
    for g in NAMES:
        rs = [r for r in rounds if r["gesture"] == g]
        if not rs:
            continue
        ms = [r["window_ms"] for r in rs]
        lat = np.array([r["latency"] for r in rs if r["latency"] is not None])
        print(f"  {NAMES[g]:6s} {len(rs)} ganger, låst i snitt {np.mean(ms):.0f} ms "
              f"etter at vinduet åpnet")
        if len(lat) > 0:
            inside = (lat <= LIMIT_MS).sum()
            print(f"         latens median {np.median(lat):.0f} ms, maks {lat.max():.0f} ms, "
                  f"{inside}/{len(lat)} innenfor {LIMIT_MS} ms")
    lat = np.array([r["latency"] for r in rounds if r["latency"] is not None])
    if len(lat) > 0:
        print(f"  Totalt: median {np.median(lat):.0f} ms, maks {lat.max():.0f} ms, "
              f"{(lat <= LIMIT_MS).sum()}/{len(lat)} innenfor {LIMIT_MS} ms")