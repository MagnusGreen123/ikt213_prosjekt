from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "negatives" / datetime.now().strftime("%Y%m%d_%H%M%S")

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720
SAVE_EVERY = 3

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Lagrer til {OUT_DIR}")
print("Mellomrom = start/pause, q = avslutt")

recording = False
frame_nr = 0
saved = 0
rejected = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame_nr += 1

    hand_found = False
    if recording:
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_found = result.multi_hand_landmarks is not None
        if hand_found:
            rejected += 1
        elif frame_nr % SAVE_EVERY == 0:
            cv2.imwrite(str(OUT_DIR / f"{saved:05d}.jpg"), frame)
            saved += 1

    view = frame.copy()
    h = view.shape[0]
    status = "RECORDING" if recording else "PAUSED"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(view, f"{status}   saved {saved}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if hand_found:
        cv2.putText(view, "HAND FOUND - not saved", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    cv2.putText(view, "space = start/pause   q = quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow("Negative bilder", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        recording = not recording
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFerdig. {saved} bilder lagret, {rejected} forkastet fordi MediaPipe fant en hånd")
if saved == 0:
    OUT_DIR.rmdir()