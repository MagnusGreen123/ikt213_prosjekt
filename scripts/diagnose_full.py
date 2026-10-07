import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PATH = ROOT / "models" / "detector_best.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_best.pt"
OUT = ROOT / "logs" / "diagnose_full.json"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720

DET_SIZE = 192
LM_SIZE = 128
PRESENCE_THRESHOLD = 0.5
TRACK_MARGIN = 0.4

SECONDS = 8
REPEATS = 2

PALM_TO_BOX = 2.95

SMOOTHING = True
MAX_SIZE_CHANGE = 0.2
BLEND = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

FINGER_NAMES = ["tommel", "peke", "lang", "ring", "lille"]
FINGER_BASE = {"tommel": 1, "peke": 5, "lang": 9, "ring": 13, "lille": 17}

BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        out = self.head(x)
        return out[:, :3], out[:, 3]


class HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        layers = []
        channels = 512
        for i in range(3):
            layers.append(nn.ConvTranspose2d(channels, 256, 4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            channels = 256
        self.upsample = nn.Sequential(*layers)
        self.head = nn.Conv2d(256, 21, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsample(x)
        heatmaps = self.head(x)

        b, k, h, w = heatmaps.shape
        prob = heatmaps.float().view(b, k, h * w).softmax(dim=2).view(b, k, h, w)

        xs = (torch.arange(w, device=x.device) + 0.5) / w
        ys = (torch.arange(h, device=x.device) + 0.5) / h
        px = (prob.sum(dim=2) * xs).sum(dim=2)
        py = (prob.sum(dim=3) * ys).sum(dim=2)

        confidence = prob.view(b, k, h * w).max(dim=2).values
        return torch.stack([px, py], dim=2).view(b, 42), confidence


detector = Detector().to(device)
detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
detector.eval()

landmarker = HeatmapModel().to(device)
landmarker.load_state_dict(torch.load(LANDMARK_PATH, map_location=device))
landmarker.eval()

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def to_tensor(img, size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    x = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return (x - mean) / std


def detect(frame):
    h, w = frame.shape[:2]
    x = to_tensor(frame, DET_SIZE)
    with torch.no_grad():
        box, presence = detector(x)
    cx = box[0, 0].item() * w
    cy = box[0, 1].item() * h
    side = box[0, 2].exp().item() * w
    p = torch.sigmoid(presence)[0].item()
    return int(cx - side / 2), int(cy - side / 2), int(side), p


def find_landmarks(frame, left, top, side):
    h, w = frame.shape[:2]
    if side < 10:
        return None, None

    crop = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None
    crop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = frame[sy0:sy1, sx0:sx1]

    x = to_tensor(crop, LM_SIZE)
    with torch.no_grad():
        pred, conf = landmarker(x)

    pts = pred.view(21, 2).cpu().numpy() * side + [left, top]
    return pts, conf.view(21).cpu().numpy()


def box_from_landmarks(pts, prev_box):
    palm = np.linalg.norm(pts[9] - pts[0])
    side = palm * PALM_TO_BOX

    cx, cy = pts.mean(axis=0)

    if prev_box is not None:
        pl, pt, ps = prev_box
        side = max(ps * 0.95, min(ps * 1.05, side))
        cx = BLEND * (pl + ps / 2) + (1 - BLEND) * cx
        cy = BLEND * (pt + ps / 2) + (1 - BLEND) * cy

    return int(cx - side / 2), int(cy - side / 2), int(side)

def finger_extension(pts):
    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-9
    return np.array([
        np.linalg.norm(pts[b + 3] - pts[b]) / scale
        for b in FINGER_BASE.values()
    ])


def run_round(name):
    mine, theirs, confs = [], [], []
    det_ms, lm_ms, mp_ms, total_ms = [], [], [], []
    boxes = []
    det_runs = 0
    collapses = 0
    frames = 0
    mp_missed = 0
    no_hand = 0

    prev_box = None

    start = time.perf_counter()
    while time.perf_counter() - start < SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frames += 1

        t_total = time.perf_counter()

        presence = 1.0
        used_det = False
        if prev_box is None:
            t0 = time.perf_counter()
            left, top, side, presence = detect(frame)
            det_ms.append((time.perf_counter() - t0) * 1000)
            det_runs += 1
            used_det = True
        else:
            left, top, side = prev_box

        pts = conf = None
        if presence >= PRESENCE_THRESHOLD:
            t0 = time.perf_counter()
            pts, conf = find_landmarks(frame, left, top, side)
            lm_ms.append((time.perf_counter() - t0) * 1000)
        else:
            no_hand += 1

        if pts is not None:
            new_box = box_from_landmarks(pts, prev_box if not used_det else None)
            bl, bt, bs = new_box
            if bs < 25 or bl + bs < 0 or bt + bs < 0 or bl > w or bt > h:
                collapses += 1
                prev_box = None
            else:
                prev_box = new_box
                boxes.append(new_box)
        else:
            prev_box = None

        total_ms.append((time.perf_counter() - t_total) * 1000)

        t0 = time.perf_counter()
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_ms.append((time.perf_counter() - t0) * 1000)

        if result.multi_hand_landmarks and pts is not None:
            lm = result.multi_hand_landmarks[0]
            mp_pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            mine.append(pts)
            theirs.append(mp_pts)
            confs.append(conf)
        elif not result.multi_hand_landmarks:
            mp_missed += 1

        if pts is not None:
            p = pts.astype(int)
            for i, (a, b) in enumerate(BONES):
                cv2.line(frame, tuple(p[a]), tuple(p[b]), (0, 255, 0), 2)
        if result.multi_hand_landmarks:
            mpp = np.array([[int(q.x * w), int(q.y * h)] for q in result.multi_hand_landmarks[0].landmark])
            for a, b in BONES:
                cv2.line(frame, tuple(mpp[a]), tuple(mpp[b]), (255, 180, 0), 1)

        cv2.rectangle(frame, (left, top), (left + side, top + side),
                      (0, 200, 255) if used_det else (150, 150, 150), 2)
        igjen = SECONDS - (time.perf_counter() - start)
        cv2.putText(frame, f"{name}  {igjen:.1f} s", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Diagnose", frame)
        cv2.waitKey(1)

    return {
        "mine": mine, "theirs": theirs, "confs": confs,
        "det_ms": det_ms, "lm_ms": lm_ms, "mp_ms": mp_ms, "total_ms": total_ms,
        "boxes": boxes, "det_runs": det_runs, "collapses": collapses,
        "frames": frames, "mp_missed": mp_missed, "no_hand": no_hand,
    }


def summarize(name, rounds):
    mine = np.array([p for r in rounds for p in r["mine"]])
    theirs = np.array([p for r in rounds for p in r["theirs"]])
    confs = np.array([c for r in rounds for c in r["confs"]])
    boxes = np.array([b for r in rounds for b in r["boxes"]])

    if len(mine) < 20:
        print(f"  {name}: for få bilder, hopper over")
        return None

    frames = sum(r["frames"] for r in rounds)
    det_runs = sum(r["det_runs"] for r in rounds)
    collapses = sum(r["collapses"] for r in rounds)
    mp_missed = sum(r["mp_missed"] for r in rounds)
    no_hand = sum(r["no_hand"] for r in rounds)

    det_ms = np.concatenate([r["det_ms"] for r in rounds]) if any(r["det_ms"] for r in rounds) else np.array([0.0])
    lm_ms = np.concatenate([r["lm_ms"] for r in rounds]) if any(r["lm_ms"] for r in rounds) else np.array([0.0])
    mp_ms = np.concatenate([r["mp_ms"] for r in rounds])
    total_ms = np.concatenate([r["total_ms"] for r in rounds])

    scale = np.linalg.norm(theirs[:, 0] - theirs[:, 9], axis=1).mean()
    diff = np.linalg.norm(mine - theirs, axis=2)

    jit_mine = np.linalg.norm(np.diff(mine, axis=0), axis=2)
    jit_theirs = np.linalg.norm(np.diff(theirs, axis=0), axis=2)

    ext_mine = np.array([finger_extension(p) for p in mine])
    ext_theirs = np.array([finger_extension(p) for p in theirs])

    result = {
        "test": name,
        "runder": len(rounds),
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
        "boks_storrelse_px": round(float(boxes[:, 2].mean()), 1) if len(boxes) else 0,
        "boks_storrelse_std": round(float(boxes[:, 2].std()), 1) if len(boxes) else 0,

        "avvik_px": round(float(diff.mean()), 2),
        "avvik_andel_av_hand": round(float(diff.mean() / scale), 3),
        "avvik_verste_punkt_px": round(float(diff.mean(axis=0).max()), 2),
        "avvik_per_finger_px": {
            FINGER_NAMES[i]: round(float(diff[:, 1 + i * 4:5 + i * 4].mean()), 2)
            for i in range(5)
        },

        "skjelving_egen_px": round(float(jit_mine.mean()), 2),
        "skjelving_mediapipe_px": round(float(jit_theirs.mean()), 2),

        "utstrekning_egen": {
            FINGER_NAMES[i]: round(float(ext_mine[:, i].mean()), 2) for i in range(5)
        },
        "utstrekning_mediapipe": {
            FINGER_NAMES[i]: round(float(ext_theirs[:, i].mean()), 2) for i in range(5)
        },
        "utstrekning_ustabilitet_egen": {
            FINGER_NAMES[i]: round(float(ext_mine[:, i].std()), 3) for i in range(5)
        },
        "utstrekning_ustabilitet_mediapipe": {
            FINGER_NAMES[i]: round(float(ext_theirs[:, i].std()), 3) for i in range(5)
        },

        "sikkerhet_snitt": round(float(confs.mean()), 4),
        "sikkerhet_per_finger": {
            FINGER_NAMES[i]: round(float(confs[:, 1 + i * 4:5 + i * 4].mean()), 4)
            for i in range(5)
        },
    }
    return result


cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

tests = [
    ("stein_rolig", "Hold knyttet neve stille"),
    ("papir_rolig", "Hold flat hånd stille"),
    ("saks_rolig", "Hold saks (peke + lang) stille"),
    ("saks_ring_lille", "Hold saks med ringfinger + lillefinger stille"),
    ("bevegelse", "Kast stein-saks-papir i normalt tempo"),
    ("avstand_naer", "Hold hånden naermere kameraet"),
    ("avstand_langt", "Hold hånden lenger unna"),
    ("flytt_rundt", "Beveg hånden rundt i hele bildet, rolig"),
    ("ingen_hand", "Ta hånden HELT ut av bildet"),
]

print(f"Sporingsglatting: {'på' if SMOOTHING else 'av'}")
print(f"{len(tests)} tester, {REPEATS} runder à {SECONDS} s\n")

results = []
for name, instruction in tests:
    print(f"=== {name} ===")
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
        no_hand = sum(x["no_hand"] for x in rounds)
        mp_missed = sum(x["mp_missed"] for x in rounds)
        res = {
            "test": name,
            "bilder": int(frames),
            "sa_ingen_hand_andel": round(no_hand / max(frames, 1), 3),
            "mediapipe_sa_ingen_hand_andel": round(mp_missed / max(frames, 1), 3),
        }
        print(json.dumps(res, indent=2, ensure_ascii=False))
        results.append(res)
        continue

    res = summarize(name, rounds)
    if res is not None:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        results.append(res)

cap.release()
cv2.destroyAllWindows()

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"smoothing": SMOOTHING, "resultater": results}, f, indent=2, ensure_ascii=False)

print(f"\nLagret til {OUT}")
print("\nOppsummering")
print(f"{'test':18s} {'avvik':>7s} {'skjelv':>7s} {'det%':>6s} {'kollaps':>8s} {'ms':>6s}")
for r in results:
    if "avvik_andel_av_hand" not in r:
        continue
    print(f"{r['test']:18s} {r['avvik_andel_av_hand']:7.3f} {r['skjelving_egen_px']:7.2f} "
          f"{r['detektor_kjort_andel'] * 100:5.0f}% {r['bokskollaps']:8d} {r['tid_total_ms']:6.1f}")
