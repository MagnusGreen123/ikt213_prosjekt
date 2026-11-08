import time
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "recordings"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720

PERSON = "NAME"                # your own name, no underscores
SESSION_THROWS = 500           # saved throws per session; None = no limit (stop with q)
COUNTDOWN = 2
RECORD_SECONDS = 2.3
PAUSE_SECONDS = 1.0

GESTURES = ["rock", "paper", "scissors"]
SAVE_IMAGES = True

MIN_CONFIDENCE = 0.5
MAX_JUMP = 0.25

NORSK = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}


hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE,
)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print("Oppløsning:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS (rapportert):", cap.get(cv2.CAP_PROP_FPS))

OUT_DIR.mkdir(parents=True, exist_ok=True)


def last_index(gesture):
    """Highest existing throw number for this gesture and person (0 if none)."""
    nums = [
        int(p.stem.rsplit("_", 1)[1])
        for p in OUT_DIR.glob(f"{gesture}_{PERSON}_*.npz")
    ]
    return max(nums, default=0)


# --- State is read from disk at startup, so the program can be stopped and
# --- restarted at any time without losing count or overwriting files.
counts = {g: last_index(g) for g in GESTURES}      # next file number base, per gesture
totals = {g: len(list(OUT_DIR.glob(f"{g}_{PERSON}_*.npz"))) for g in GESTURES}
total_all = len(list(OUT_DIR.glob("*.npz")))       # every throw in the dataset
saved = []                                         # throws saved in THIS session
bag = []                                           # shuffle bag for gesture order


def status_lines():
    limit = f"/{SESSION_THROWS}" if SESSION_THROWS else ""
    return [
        f"Økt: {len(saved)}{limit} lagret",
        f"Totalt lagret: {total_all} ({PERSON}: {sum(totals.values())})",
    ]


def next_gesture():
    """Random order, balanced within every bag of 30."""
    if not bag:
        bag.extend(random.sample(GESTURES * 10, 30))
    return bag.pop()


def read_frame():
    ok, frame = cap.read()
    if not ok:
        return None, None
    return cv2.flip(frame, 1), time.perf_counter()


def detect(frame):
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None
    hand = result.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)


def draw_landmarks(frame, points):
    h, w = frame.shape[:2]
    for x, y, z in points:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)


def show(frame, lines, color=(0, 255, 0)):
    y = 40
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y += 40
    cv2.imshow("Opptak", frame)


def big_text(frame, text, color):
    h, w = frame.shape[:2]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
    pos = ((w - size[0]) // 2, (h + size[1]) // 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 8)


def countdown(gesture):
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= COUNTDOWN:
            return True

        frame, _ = read_frame()
        if frame is None:
            return False

        igjen = int(COUNTDOWN - elapsed) + 1
        big_text(frame, str(igjen), (0, 200, 255))
        show(frame, status_lines() + [
            f"Neste: {NORSK[gesture]}",
            "Start med knyttet hand",
            "p = pause, q = avslutt",
        ])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

        if key == ord("p"):
            # Pause: keep reading frames (so the camera buffer stays fresh)
            # until p is pressed again, then restart the countdown.
            while True:
                frame, _ = read_frame()
                if frame is None:
                    return False
                big_text(frame, "PAUSE", (0, 200, 255))
                show(frame, status_lines() + ["p = fortsett"], (0, 200, 255))
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    break
            start = time.perf_counter()


def record_throw(gesture):
    points_list = []
    times = []
    images = []

    total_frames = 0
    missed = 0

    start = time.perf_counter()
    while time.perf_counter() - start < RECORD_SECONDS:
        frame, t = read_frame()
        if frame is None:
            break

        total_frames += 1
        points = detect(frame)

        if points is None:
            missed += 1
        else:
            points_list.append(points)
            times.append(t)
            if SAVE_IMAGES:
                images.append(frame.copy())
            draw_landmarks(frame, points)

        big_text(frame, NORSK[gesture], (0, 0, 255))
        show(frame, status_lines())
        cv2.waitKey(1)

    if not times:
        return None

    return np.array(points_list), np.array(times) - times[0], images, total_frames, missed


def filter_jumps(points, times, images):
    keep = [0]
    for i in range(1, len(points)):
        dt = times[i] - times[i - 1]
        step = np.linalg.norm(points[i][:, :2] - points[i - 1][:, :2], axis=1).max()
        if step < 200.0 * dt:
            keep.append(i)

    images_kept = [images[i] for i in keep] if images else []
    return points[keep], times[keep], images_kept, len(points) - len(keep)


def save_throw(number, gesture, points, times, images):
    # Never overwrite: if this number is taken, move to the next free one.
    while (OUT_DIR / f"{gesture}_{PERSON}_{number:03d}.npz").exists():
        number += 1

    name = f"{gesture}_{PERSON}_{number:03d}"
    np.savez(
        OUT_DIR / f"{name}.npz",
        landmarks=points,
        times=times,
        label=gesture,
        person=PERSON,
    )

    if images:
        img_dir = OUT_DIR / name
        if img_dir.exists():
            shutil.rmtree(img_dir)   # remove leftover frames from an older run
        img_dir.mkdir()
        for i, img in enumerate(images):
            cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), img)

    return name


def delete_throw(name):
    npz = OUT_DIR / f"{name}.npz"
    if npz.exists():
        npz.unlink()
    img_dir = OUT_DIR / name
    if img_dir.exists():
        shutil.rmtree(img_dir)


print(f"\nPerson: '{PERSON}'")
print("Allerede lagret: " + ", ".join(f"{g} {totals[g]}" for g in GESTURES))
print(f"Totalt i datasettet: {total_all}")
if SESSION_THROWS:
    print(f"Mål denne økten: {SESSION_THROWS} lagrede kast")
else:
    print("Ingen grense denne økten (avslutt med q)")
print("Start med knyttet hand, pump 2-3 ganger, kast i normalt spilltempo.")
print("Mellomrom = start, u = slett forrige kast, p = pause, q = avslutt\n")

while True:
    frame, _ = read_frame()
    if frame is None:
        break
    show(frame, ["Mellomrom = start opptak", "q = avslutt"] + status_lines())
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        break
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

attempt = 0

while SESSION_THROWS is None or len(saved) < SESSION_THROWS:
    gesture = next_gesture()
    if not countdown(gesture):
        break

    attempt += 1
    result = record_throw(gesture)
    if result is None:
        print(f"Forsøk {attempt}: ingen hånd funnet i det hele tatt, hopper over")
        continue

    points, times, images, total_frames, missed = result
    points, times, images, dropped = filter_jumps(points, times, images)

    steps = np.diff(times)
    fps = len(times) / times[-1] if times[-1] > 0 else 0

    print(f"Forsøk {attempt} [{gesture}]:")
    print(f"  kamera: {total_frames} bilder, MediaPipe mistet handen i {missed}")
    print(f"  filter forkastet: {dropped}, beholdt: {len(points)}")
    print(f"  effektiv fps: {fps:.1f}")
    if len(steps) > 0:
        print(f"  tidssteg: median {np.median(steps)*1000:.0f} ms, "
              f"maks {steps.max()*1000:.0f} ms")
        if steps.max() > 0.1:
            print(f"  ADVARSEL: hull på {steps.max()*1000:.0f} ms midt i kastet")

    if len(points) < 20:
        print("  ADVARSEL: få bilder, sjekk lys og at handen er synlig")

    name = save_throw(counts[gesture] + 1, gesture, points, times, images)
    counts[gesture] = int(name.rsplit("_", 1)[1])
    totals[gesture] += 1
    total_all += 1
    saved.append(name)

    start = time.perf_counter()
    while time.perf_counter() - start < PAUSE_SECONDS:
        frame, _ = read_frame()
        if frame is None:
            break
        show(frame, [f"Lagret {name}: {len(points)} bilder"] + status_lines()
             + ["u = angre dette kastet"])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("u") and saved:
            delete_throw(saved.pop())
            counts[gesture] = last_index(gesture)
            totals[gesture] -= 1
            total_all -= 1
            print("  Angret, kastet er slettet")
            break

cap.release()
cv2.destroyAllWindows()

print(f"\nFerdig. {len(saved)} kast lagret denne økten i {OUT_DIR}")
print(f"Totalt i datasettet: {total_all}")
for g in GESTURES:
    print(f"  {g}: {totals[g]} ({PERSON})")