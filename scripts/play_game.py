

import sys
import time
import math
import random
from collections import deque
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline
from src.classification import gesture_2 as G
from src.classification.snapshot import save_lock_snapshot

try:
    import winsound
except ImportError:
    winsound = None

DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

CAMERA_ID = 0
CAM_W = 1280
CAM_H = 720
PANEL_W = 440
ICON = 95
SHOW_RESULT = 2.5
WIN_SCORE = 2
SOUND = True
UNBEATABLE = True
SAVE_LOCKS = True
LOCK_FRAMES = 2
WINDOW = "ROCK PAPER SCISSORS"

MOVES = ["rock", "paper", "scissors"]
NAMES = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}
NORSK = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

BG = (22, 16, 12)
PANEL = (34, 25, 18)
CARD = (52, 40, 30)
WHITE = (245, 245, 245)
MUTED = (150, 135, 120)
BLUE = (255, 170, 40)
PINK = (190, 70, 255)
GREEN = (110, 235, 90)
RED = (80, 70, 255)
YELLOW = (40, 215, 255)
ORANGE = (40, 140, 255)
FINGER_COLORS = [YELLOW, BLUE, GREEN, PINK, ORANGE]
GESTURE_COLOR = {"rock": PINK, "paper": BLUE, "scissors": YELLOW}
RESULT_STYLE = {"player": ("YOU WIN", GREEN), "computer": ("CPU WINS", RED), "draw": ("DRAW", YELLOW)}

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BIG = cv2.FONT_HERSHEY_TRIPLEX

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


# ---------------- spillregler ----------------
def winner(player, computer):
    if player == computer:
        return "draw"
    if BEATS[player] == computer:
        return "player"
    return "computer"


def check_rules():
    expected = {
        ("rock", "rock"): "draw", ("rock", "paper"): "computer", ("rock", "scissors"): "player",
        ("paper", "rock"): "player", ("paper", "paper"): "draw", ("paper", "scissors"): "computer",
        ("scissors", "rock"): "computer", ("scissors", "paper"): "player", ("scissors", "scissors"): "draw",
    }
    right = 0
    for (p, c), want in expected.items():
        got = winner(p, c)
        if got == want:
            right += 1
        else:
            print(f"  FEIL: {p} mot {c} ga {got}, skulle vært {want}")
    print(f"Regelsjekk: {right}/9 kombinasjoner riktige")


def beep(kind):
    if SOUND and winsound is not None:
        sounds = {"win": winsound.MB_OK, "lose": winsound.MB_ICONHAND,
                  "draw": winsound.MB_ICONASTERISK, "throw": winsound.MB_ICONEXCLAMATION}
        winsound.MessageBeep(sounds[kind])


# ---------------- tegnehjelpere ----------------
def dim(color, f=0.35):
    return tuple(int(c * f) for c in color)


def text(img, s, x, y, scale, color, thick=2, font=FONT, center=False, outline=True):
    size = cv2.getTextSize(s, font, scale, thick)[0]
    if center:
        x -= size[0] // 2
    if outline:
        cv2.putText(img, s, (x, y), font, scale, (0, 0, 0), thick + 5, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return size


def rounded_rect(img, x1, y1, x2, y2, r, color, filled=True, thick=2):
    if filled:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thick, cv2.LINE_AA)


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    over = np.full_like(roi, color)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, roi)


def draw_icon(img, gesture, cx, cy, size, color):
    fingers = {
        "paper": [(-32, 0.55), (-11, 0.62), (10, 0.6), (30, 0.5)],
        "rock": [(-30, 0.13), (-10, 0.15), (10, 0.15), (30, 0.13)],
        "scissors": [(-30, 0.62), (2, 0.64), (18, 0.14), (34, 0.12)],
    }[gesture]
    thumb = (-80, 0.42) if gesture == "paper" else (-70, 0.16)
    palm_r = int(size * 0.30)
    thick = max(4, int(size * 0.14))
    for layer, (col, extra) in enumerate([(dim(color, 0.3), 10), (color, 0)]):
        cv2.circle(img, (cx, cy), palm_r + extra // 2, col, -1, cv2.LINE_AA)
        for angle, length in fingers + [thumb]:
            a = math.radians(angle)
            bx = cx + int(math.sin(a) * palm_r * 0.75)
            by = cy - int(math.cos(a) * palm_r * 0.75)
            tx = cx + int(math.sin(a) * (palm_r * 0.75 + size * length))
            ty = cy - int(math.cos(a) * (palm_r * 0.75 + size * length))
            cv2.line(img, (bx, by), (tx, ty), col, thick + extra, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), (thick + extra) // 2, col, -1, cv2.LINE_AA)


def draw_skeleton(img, pts):
    p = [(int(x), int(y)) for x, y in pts]
    for a, b in CONNECTIONS:
        if (a, b) in [(5, 9), (9, 13), (13, 17)]:
            col = MUTED
        else:
            col = FINGER_COLORS[(b - 1) // 4]
        cv2.line(img, p[a], p[b], dim(col, 0.45), 10, cv2.LINE_AA)
        cv2.line(img, p[a], p[b], col, 3, cv2.LINE_AA)
    for i, pt in enumerate(p):
        cv2.circle(img, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 4, WHITE if i % 4 == 0 else FINGER_COLORS[max(0, (i - 1) // 4)], -1, cv2.LINE_AA)


def draw_brackets(img, pts, color):
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    pad = 25
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    L = max(20, (x2 - x1) // 5)
    for (x, y, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * L, y), color, 4, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * L), color, 4, cv2.LINE_AA)


# ---------------- konfetti ----------------
confetti = []


def spawn_confetti(n=180):
    for _ in range(n):
        confetti.append([random.uniform(0, CAM_W), random.uniform(-300, 0),
                         random.uniform(-2, 2), random.uniform(2, 7),
                         random.choice([BLUE, PINK, GREEN, YELLOW, ORANGE, WHITE]),
                         random.randint(5, 10)])


def update_confetti(img):
    for c in confetti:
        c[0] += c[2]
        c[1] += c[3]
        c[3] += 0.12
        x, y, s = int(c[0]), int(c[1]), c[5]
        cv2.rectangle(img, (x, y), (x + s, y + s // 2), c[4], -1)
    confetti[:] = [c for c in confetti if c[1] < CAM_H + 20]


# ---------------- sidepanel ----------------
def draw_panel(canvas, t, state):
    x0 = CAM_W
    canvas[:, x0:] = PANEL
    cx = x0 + PANEL_W // 2

    if UNBEATABLE:
        text(canvas, "UNBEATABLE  MODE", cx, 45, 0.75, RED, 2, center=True, outline=False)
    else:
        text(canvas, "ROCK  PAPER  SCISSORS", cx, 45, 0.75, BLUE, 2, center=True, outline=False)
    cv2.line(canvas, (x0 + 30, 62), (x0 + PANEL_W - 30, 62), CARD, 2)


    text(canvas, "YOU", x0 + 95, 105, 0.7, MUTED, 2, center=True, outline=False)
    text(canvas, "CPU", x0 + PANEL_W - 95, 105, 0.7, MUTED, 2, center=True, outline=False)
    text(canvas, str(state["score"]["player"]), x0 + 95, 185, 3.0, GREEN, 6, FONT_BIG, center=True, outline=False)
    text(canvas, str(state["score"]["computer"]), x0 + PANEL_W - 95, 185, 3.0, RED, 6, FONT_BIG,
         center=True, outline=False)
    text(canvas, "-", cx, 170, 2.0, MUTED, 4, FONT_BIG, center=True, outline=False)
    text(canvas, f"first to {WIN_SCORE}", cx, 215, 0.55, MUTED, 1, center=True, outline=False)


    card_y1, card_y2 = 235, 460
    for side, x1 in [("player", x0 + 25), ("computer", x0 + PANEL_W // 2 + 8)]:
        x2 = x1 + PANEL_W // 2 - 33
        rounded_rect(canvas, x1, card_y1, x2, card_y2, 18, CARD)
        icx, icy = (x1 + x2) // 2, card_y1 + 125
        result = state["result"]
        if side == "player":
            label = "YOU"
            if result is not None:
                g = result[0]
                draw_icon(canvas, g, icx, icy, ICON, GESTURE_COLOR[g])
                text(canvas, NAMES[g], icx, card_y2 - 25, 0.7, WHITE, 2, center=True, outline=False)
            elif state["live_gesture"] is not None:
                g = state["live_gesture"]
                draw_icon(canvas, g, icx, icy, ICON, dim(GESTURE_COLOR[g], 0.55))
                text(canvas, NAMES[g], icx, card_y2 - 25, 0.7, MUTED, 2, center=True, outline=False)
            else:
                text(canvas, "...", icx, icy + 20, 2.0, MUTED, 3, FONT_BIG, center=True, outline=False)
        else:
            label = "CPU"
            if result is not None:
                g = result[1]
                pop = min(1.0, 0.5 + (t - state["result_time"]) / 0.2 * 0.5)
                draw_icon(canvas, g, icx, icy, int(ICON * pop), GESTURE_COLOR[g])
                text(canvas, NAMES[g], icx, card_y2 - 25, 0.7, WHITE, 2, center=True, outline=False)
            elif state["window_open"]:
                g = MOVES[int(t * 14) % 3]
                draw_icon(canvas, g, icx, icy, ICON, dim(GESTURE_COLOR[g], 0.5))
                text(canvas, "READY", icx, card_y2 - 25, 0.7, YELLOW, 2, center=True, outline=False)
            else:
                text(canvas, "?", icx, icy + 30, 3.0, MUTED, 5, FONT_BIG, center=True, outline=False)
                text(canvas, "watching" if UNBEATABLE else "locked in", icx, card_y2 - 25, 0.6, MUTED, 1,
                     center=True, outline=False)
        text(canvas, label, icx, card_y1 + 30, 0.6, MUTED, 1, center=True, outline=False)
        if result is not None:
            who = result[2]
            if (who == side) or who == "draw":
                rounded_rect(canvas, x1, card_y1, x2, card_y2, 18, RESULT_STYLE[who][1], filled=False, thick=3)
    text(canvas, "VS", cx, (card_y1 + card_y2) // 2 + 12, 0.9, WHITE, 2, FONT_BIG, center=True)

    y = 490
    text(canvas, "CONFIDENCE", x0 + 30, y, 0.5, MUTED, 1, outline=False)
    rounded_rect(canvas, x0 + 30, y + 12, x0 + PANEL_W - 30, y + 28, 8, CARD)
    frac = min(1.0, state["confidence"] / 0.3)
    col = GREEN if state["confidence"] >= G.MARGIN else YELLOW
    if frac > 0.05:
        rounded_rect(canvas, x0 + 30, y + 12, x0 + 30 + int((PANEL_W - 60) * frac), y + 28, 8, col)
    mx = x0 + 30 + int((PANEL_W - 60) * G.MARGIN / 0.3)
    cv2.line(canvas, (mx, y + 8), (mx, y + 32), WHITE, 2)

    y = 560
    text(canvas, "LAST ROUNDS", x0 + 30, y, 0.5, MUTED, 1, outline=False)
    for i, who in enumerate(state["history"][-8:]):
        col = {"player": GREEN, "computer": RED, "draw": YELLOW}[who]
        letter = {"player": "W", "computer": "L", "draw": "D"}[who]
        px = x0 + 50 + i * 47
        cv2.circle(canvas, (px, y + 32), 18, col, -1, cv2.LINE_AA)
        text(canvas, letter, px, y + 40, 0.6, (0, 0, 0), 2, center=True, outline=False)
    if state["streak"] >= 2:
        text(canvas, f"WIN STREAK x{state['streak']}", x0 + PANEL_W - 30 - 190, y, 0.5, ORANGE, 1,
             outline=False)

    text(canvas, f"{state['fps']:.0f} fps   {state['ms']:.0f} ms/frame", x0 + 30, CAM_H - 45, 0.5, MUTED, 1,
         outline=False)
    text(canvas, "r round  n match  c mode  f full  s sound  q quit", x0 + 30, CAM_H - 20, 0.45, MUTED, 1,
         outline=False)


# ---------------- kameradelen ----------------
def draw_camera_overlay(img, t, state, decider, pts):
    blend_rect(img, 0, 0, CAM_W, 90, (0, 0, 0), 0.45)

    if pts is not None:
        col = GESTURE_COLOR[state["live_gesture"]] if state["live_gesture"] else MUTED
        draw_brackets(img, pts, col)
        draw_skeleton(img, pts)
    else:
        text(img, "SHOW YOUR HAND", CAM_W // 2, CAM_H - 40, 1.0, MUTED, 2, center=True)

    result = state["result"]
    if result is not None:
        who = result[2]
        label, col = RESULT_STYLE[who]
        age = t - state["result_time"]
        flash = max(0.0, 0.45 - age * 1.2)
        if flash > 0:
            blend_rect(img, 0, 0, CAM_W, CAM_H, col, flash)
        pop = min(1.0, 0.6 + age / 0.25 * 0.4)
        blend_rect(img, 0, CAM_H // 2 - 110, CAM_W, CAM_H // 2 + 70, (0, 0, 0), 0.55)
        text(img, label, CAM_W // 2, CAM_H // 2 + 10, 3.2 * pop, col, int(9 * pop), FONT_BIG, center=True)
        sub = f"{NAMES[result[0]]}  vs  {NAMES[result[1]]}"
        text(img, sub, CAM_W // 2, CAM_H // 2 + 55, 1.0, WHITE, 2, center=True)
    elif state["match_over"]:
        blend_rect(img, 0, 0, CAM_W, CAM_H, (0, 0, 0), 0.55)
        won = state["score"]["player"] > state["score"]["computer"]
        label, col = ("CHAMPION", GREEN) if won else ("DEFEATED", RED)
        s = 1.0 + 0.04 * math.sin(t * 6)
        text(img, label, CAM_W // 2, CAM_H // 2, 3.6 * s, col, 10, FONT_BIG, center=True)
        text(img, f"{state['score']['player']} - {state['score']['computer']}", CAM_W // 2, CAM_H // 2 + 70,
             1.6, WHITE, 3, FONT_BIG, center=True)
        text(img, "press N for a new match", CAM_W // 2, CAM_H // 2 + 125, 0.9, MUTED, 2, center=True)
    elif decider.window_open:
        s = 1.0 + 0.12 * abs(math.sin(t * 10))
        text(img, "THROW!", CAM_W // 2, 70, 2.0 * s, YELLOW, 6, FONT_BIG, center=True)
    else:
        n = decider.pumps_needed
        for i in range(n):
            px = CAM_W // 2 - (n - 1) * 45 + i * 90
            filled = i < decider.pumps
            cv2.circle(img, (px, 45), 22, BLUE if filled else MUTED, -1 if filled else 3, cv2.LINE_AA)
        text(img, "PUMP", CAM_W // 2 - (n - 1) * 45 - 150, 58, 1.0, WHITE, 2, center=True)

    update_confetti(img)


# ---------------- hovedløkke ----------------
check_rules()

pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
decider = G.ThrowDecider()
print("Enhet:", pipe.device)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, (CAM_W + PANEL_W) * 3 // 4, CAM_H * 3 // 4)
fullscreen = False

state = {"score": {"player": 0, "computer": 0}, "result": None, "result_time": 0.0,
         "match_over": False, "history": [], "streak": 0, "live_gesture": None,
         "confidence": 0.0, "window_open": False, "fps": 0.0, "ms": 0.0}
computer = random.choice(MOVES)
rounds = 0
start_time = time.perf_counter()
last_t = start_time
frames = 0
was_open = False
recent = deque(maxlen=LOCK_FRAMES)

print(f"Pump {decider.pumps_needed} ganger og kast. Første til {WIN_SCORE} vinner. "
      f"Maskinen er {'uslåelig' if UNBEATABLE else 'rettferdig'} (c bytter).")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
        frame = cv2.resize(frame, (CAM_W, CAM_H))
    t = time.perf_counter()
    frames += 1
    dt = t - last_t
    last_t = t
    if dt > 0:
        state["fps"] = 1 / dt if state["fps"] == 0 else 0.9 * state["fps"] + 0.1 / dt

    pts, conf = pipe.process(frame)
    state["ms"] = pipe.last.get("total_ms", 0)
    recent.append((frame.copy(), None if pts is None else np.asarray(pts, dtype=float)))
    if pts is not None:
        pts = np.asarray(pts, dtype=float)
        g, c = G.classify_frame(pts)
        state["live_gesture"], state["confidence"] = g, c
    else:
        state["live_gesture"], state["confidence"] = None, 0.0

    if not state["match_over"] or state["result"] is not None:
        if state["result"] is None:
            still_now = (pts is not None and decider.prev_ext is not None and
                         np.max(np.abs(G.finger_extension(pts)[1:] - decider.prev_ext)) < G.SHAPE_STILL)
            decision = decider.update(pts, t)
            recent[-1] = (recent[-1][0], recent[-1][1],
                          {"window": decider.window_open, "streak": decider.streak,
                           "pumps": decider.pumps, "still": bool(still_now)})
            if decider.window_open and not was_open:
                beep("throw")
            was_open = decider.window_open

            if decision is not None:
                if UNBEATABLE:
                    computer = COUNTER[decision]
                who = winner(decision, computer)
                if who != "draw":
                    state["score"][who] += 1
                state["streak"] = state["streak"] + 1 if who == "player" else 0
                state["history"].append(who)
                state["result"] = (decision, computer, who)
                state["result_time"] = t
                rounds += 1
                if SAVE_LOCKS:
                    path = save_lock_snapshot(ROOT / "results" / "locks", rounds, decision, list(recent))
                    print(f"  lagret {path.name}")
                beep({"player": "win", "computer": "lose", "draw": "draw"}[who])
                print(f"Runde {rounds}: du {NORSK[decision]:6s} maskinen {NORSK[computer]:6s} -> "
                      f"{ {'player': 'du vant', 'computer': 'maskinen vant', 'draw': 'uavgjort'}[who] }   "
                      f"stilling {state['score']['player']}-{state['score']['computer']}")
                if max(state["score"].values()) >= WIN_SCORE:
                    state["match_over"] = True

        elif t - state["result_time"] > SHOW_RESULT:
            decider.reset()
            was_open = False
            state["result"] = None
            computer = random.choice(MOVES)
            if state["match_over"]:
                won = state["score"]["player"] > state["score"]["computer"]
                print("Kampen er ferdig:", "du vant" if won else "maskinen vant")
                if won:
                    spawn_confetti()

    state["window_open"] = decider.window_open and state["result"] is None

    canvas = np.zeros((CAM_H, CAM_W + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :] = BG
    draw_camera_overlay(frame, t, state, decider, pts)
    canvas[:, :CAM_W] = frame
    draw_panel(canvas, t, state)
    cv2.imshow(WINDOW, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        decider.reset()
        was_open = False
        state["result"] = None
        computer = random.choice(MOVES)
    elif key == ord("n"):
        decider.reset()
        was_open = False
        state["result"] = None
        state["score"] = {"player": 0, "computer": 0}
        state["match_over"] = False
        state["streak"] = 0
        confetti.clear()
        computer = random.choice(MOVES)
        print("Ny kamp")
    elif key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord("c"):
        UNBEATABLE = not UNBEATABLE
        decider.reset()
        was_open = False
        state["result"] = None
        computer = random.choice(MOVES)
        print("Maskinen er nå", "uslåelig" if UNBEATABLE else "rettferdig")
    elif key == ord("s"):
        SOUND = not SOUND
        print("Lyd", "på" if SOUND else "av")

end_time = time.perf_counter()
cap.release()
cv2.destroyAllWindows()

seconds = end_time - start_time
print(f"\nSpilte i {seconds:.0f} s, {frames} bilder, snitt {frames / seconds:.1f} fps, {rounds} runder")
if seconds < 60:
    print("Under 60 s: spill lenger for å teste FR-11")