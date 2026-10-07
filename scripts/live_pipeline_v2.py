import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline, draw_hand, draw_box

DETECTOR_PATH = ROOT / "models" / "detector_palm.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei.pt"
CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720

pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
print("Enhet:", pipe.device)
print("q = avslutt, d = skru sporing av/på, r = nullstill sporing")

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

det_ms, lm_ms, total_ms, jitter = [], [], [], []
box_sizes = []
prev_pts = None
det_runs = 0
collapses = 0
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frames += 1

    pts, conf = pipe.process(frame)
    info = pipe.last

    if pts is not None and prev_pts is not None:
        jitter.append(float(np.linalg.norm(pts - prev_pts, axis=1).mean()))
    prev_pts = pts

    if info["det_ms"] is not None:
        det_ms.append(info["det_ms"])
        det_runs += 1
    if info["lm_ms"] is not None:
        lm_ms.append(info["lm_ms"])
    total_ms.append(info["total_ms"])
    if info["collapsed"]:
        collapses += 1
    if pts is not None:
        box_sizes.append(info["box"][2])

    if pts is not None:
        draw_hand(frame, pts)
    draw_box(frame, info["box"], (0, 200, 255) if info["used_detector"] else (150, 150, 150))

    lines = [
        f"Total: {np.mean(total_ms[-30:]):.1f} ms",
        f"Detektor: {np.mean(det_ms[-30:]):.1f} ms" if det_ms else "Detektor: -",
        f"Landmarks: {np.mean(lm_ms[-30:]):.1f} ms" if lm_ms else "Landmarks: -",
        f"Skjelving: {np.mean(jitter[-30:]):.2f} px" if jitter else "Skjelving: -",
        f"Boks: {np.mean(box_sizes[-30:]):.0f} px" if box_sizes else "Boks: -",
        f"Detektor kjort: {det_runs}/{frames}",
        f"Bokskollaps: {collapses}",
        f"Sporing: {'paa' if pipe.tracking else 'av'}",
    ]
    if info["presence"] < 0.5:
        lines.append(f"Ingen hand ({info['presence']:.2f})")

    y = 35
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    cv2.imshow("Egen pipeline", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):
        pipe.tracking = not pipe.tracking
        pipe.reset()
        prev_pts = None
    if key == ord("r"):
        pipe.reset()
        prev_pts = None

cap.release()
cv2.destroyAllWindows()

print(f"\nBilder: {frames}")
print(f"Detektor kjørt: {det_runs} ({det_runs / max(frames, 1) * 100:.1f}%)")
print(f"Bokskollaps: {collapses}")
if jitter:
    print(f"Skjelving: {np.mean(jitter):.2f} px snitt, {np.median(jitter):.2f} px median")
if box_sizes:
    print(f"Boksstørrelse: {np.mean(box_sizes):.0f} px snitt, std {np.std(box_sizes):.0f}")
if det_ms:
    print(f"Detektor: {np.mean(det_ms):.1f} ms")
if lm_ms:
    print(f"Landmarks: {np.mean(lm_ms):.1f} ms")
print(f"Totalt: {np.mean(total_ms):.1f} ms snitt, {np.percentile(total_ms, 95):.1f} ms p95")
