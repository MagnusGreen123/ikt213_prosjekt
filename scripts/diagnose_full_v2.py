import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.pipeline import HandPipeline, draw_hand, draw_box

DETECTOR_PATH = ROOT / "models" / "detector_palm.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm.pt"
OUT = ROOT / "logs" / "diagnose_full.json"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720
SECONDS = 8
REPEATS = 2

FINGER_NAMES = ["tommel", "peke", "lang", "ring", "lille"]
FINGER_BASE = [1, 5, 9, 13, 17]

pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking = False)
print("Enhet:", pipe.device)

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def finger_extension(pts):
    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-9
    return np.array([np.linalg.norm(pts[b + 3] - pts[b]) / scale for b in FINGER_BASE])


def run_round(name):
    mine, theirs, confs, boxes = [], [], [], []
    det_ms, lm_ms, mp_ms, total_ms = [], [], [], []
    det_runs = collapses = frames = mp_missed = no_hand = 0

    pipe.reset()
    start = time.perf_counter()

    while time.perf_counter() - start < SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frames += 1

        pts, conf = pipe.process(frame)
        info = pipe.last

        if info["det_ms"] is not None:
            det_ms.append(info["det_ms"])
            det_runs += 1
        if info["lm_ms"] is not None:
            lm_ms.append(info["lm_ms"])
        total_ms.append(info["total_ms"])
        if info["collapsed"]:
            collapses += 1
        if info["presence"] < 0.5:
            no_hand += 1
        if pts is not None:
            boxes.append(info["box"])

        t0 = time.perf_counter()
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_ms.append((time.perf_counter() - t0) * 1000)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            mp_pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            draw_hand(frame, mp_pts, color=(255, 180, 0), dots=False)
            if pts is not None:
                mine.append(pts)
                theirs.append(mp_pts)
                confs.append(conf)
        else:
            mp_missed += 1

        if pts is not None:
            draw_hand(frame, pts, color=(0, 255, 0))
        draw_box(frame, info["box"], (0, 200, 255) if info["used_detector"] else (150, 150, 150))

        igjen = SECONDS - (time.perf_counter() - start)
        cv2.putText(frame, f"{name}  {igjen:.1f} s", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "gronn = egen, blaa = MediaPipe", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Diagnose", frame)
        cv2.waitKey(1)

    return dict(mine=mine, theirs=theirs, confs=confs, boxes=boxes,
                det_ms=det_ms, lm_ms=lm_ms, mp_ms=mp_ms, total_ms=total_ms,
                det_runs=det_runs, collapses=collapses, frames=frames,
                mp_missed=mp_missed, no_hand=no_hand)


def summarize(name, rounds):
    mine = np.array([p for r in rounds for p in r["mine"]])
    theirs = np.array([p for r in rounds for p in r["theirs"]])
    confs = np.array([c for r in rounds for c in r["confs"]])
    boxes = np.array([b for r in rounds for b in r["boxes"]])

    if len(mine) < 20:
        print("  for få bilder")
        return None

    frames = sum(r["frames"] for r in rounds)
    det_runs = sum(r["det_runs"] for r in rounds)
    collapses = sum(r["collapses"] for r in rounds)
    mp_missed = sum(r["mp_missed"] for r in rounds)
    no_hand = sum(r["no_hand"] for r in rounds)

    def cat(key):
        vals = [v for r in rounds for v in r[key]]
        return np.array(vals) if vals else np.array([0.0])

    det_ms, lm_ms, mp_ms, total_ms = cat("det_ms"), cat("lm_ms"), cat("mp_ms"), cat("total_ms")

    scale = np.linalg.norm(theirs[:, 0] - theirs[:, 9], axis=1).mean()
    diff = np.linalg.norm(mine - theirs, axis=2)
    jit_mine = np.linalg.norm(np.diff(mine, axis=0), axis=2)
    jit_theirs = np.linalg.norm(np.diff(theirs, axis=0), axis=2)

    ext_mine = np.array([finger_extension(p) for p in mine])
    ext_theirs = np.array([finger_extension(p) for p in theirs])

    return {
        "test": name,
        "bilder": int(frames),
        "bilder_med_begge": int(len(mine)),
        "tid_total_ms": round(float(total_ms.mean()), 1),
        "tid_total_p95_ms": round(float(np.percentile(total_ms, 95)), 1),
        "tid_detektor_ms": round(float(det_ms.mean()), 1),
        "tid_landmarks_ms": round(float(lm_ms.mean()), 1),
        "tid_mediapipe_ms": round(float(mp_ms.mean()), 1),
        "detektor_kjort_andel": round(det_runs / max(frames, 1), 3),
        "bokskollaps": int(collapses),
        "ingen_hand_funnet": int(no_hand),
        "mediapipe_mistet_hand": int(mp_missed),
        "boks_storrelse_px": round(float(boxes[:, 2].mean()), 1),
        "boks_storrelse_std": round(float(boxes[:, 2].std()), 1),
        "avvik_px": round(float(diff.mean()), 2),
        "avvik_andel_av_hand": round(float(diff.mean() / scale), 3),
        "avvik_verste_punkt_px": round(float(diff.mean(axis=0).max()), 2),
        "avvik_per_finger_px": {
            FINGER_NAMES[i]: round(float(diff[:, 1 + i * 4:5 + i * 4].mean()), 2) for i in range(5)
        },
        "skjelving_egen_px": round(float(jit_mine.mean()), 2),
        "skjelving_mediapipe_px": round(float(jit_theirs.mean()), 2),
        "utstrekning_egen": {
            FINGER_NAMES[i]: round(float(ext_mine[:, i].mean()), 2) for i in range(5)
        },
        "utstrekning_mediapipe": {
            FINGER_NAMES[i]: round(float(ext_theirs[:, i].mean()), 2) for i in range(5)
        },
        "utstrekning_feil": {
            FINGER_NAMES[i]: round(float(np.abs(ext_mine[:, i] - ext_theirs[:, i]).mean()), 3)
            for i in range(5)
        },
        "sikkerhet_snitt": round(float(confs.mean()), 4),
    }


tests = [
    ("stein_rolig", "Hold knyttet neve stille"),
    ("papir_rolig", "Hold flat haand stille"),
    ("saks_rolig", "Hold saks (peke + lang) stille"),
    ("saks_ring_lille", "Hold saks med ringfinger + lillefinger stille"),
    ("bevegelse", "Kast stein-saks-papir i normalt tempo"),
    ("avstand_naer", "Hold haanden naermere kameraet"),
    ("avstand_langt", "Hold haanden lenger unna"),
    ("flytt_rundt", "Beveg handen rundt i hele bildet, rolig"),
    ("ingen_hand", "Ta handen HELT ut av bildet"),
]

results = []
for name, instruction in tests:
    print(f"\n=== {name} ===")
    print(instruction)

    skip = False
    while True:
        ok, frame = cap.read()
        if not ok:
            skip = True
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, name, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, instruction[:40], (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "mellomrom = start, q = hopp over", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Diagnose", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            skip = True
            break
    if skip:
        continue

    rounds = []
    for r in range(REPEATS):
        print(f"  runde {r + 1}/{REPEATS}")
        rounds.append(run_round(name))
        time.sleep(0.5)

    if name == "ingen_hand":
        frames = sum(x["frames"] for x in rounds)
        res = {
            "test": name,
            "bilder": int(frames),
            "sa_ingen_hand_andel": round(sum(x["no_hand"] for x in rounds) / max(frames, 1), 3),
            "mediapipe_sa_ingen_hand_andel": round(
                sum(x["mp_missed"] for x in rounds) / max(frames, 1), 3),
        }
    else:
        res = summarize(name, rounds)

    if res is not None:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        results.append(res)

cap.release()
cv2.destroyAllWindows()

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nLagret til {OUT}")
print("\nOppsummering")
print(f"{'test':18s} {'avvik':>7s} {'skjelv':>7s} {'det%':>6s} {'kollaps':>8s} {'ms':>6s}")
for r in results:
    if "avvik_andel_av_hand" not in r:
        continue
    print(f"{r['test']:18s} {r['avvik_andel_av_hand']:7.3f} {r['skjelving_egen_px']:7.2f} "
          f"{r['detektor_kjort_andel'] * 100:5.1f}% {r['bokskollaps']:8d} {r['tid_total_ms']:6.1f}")
