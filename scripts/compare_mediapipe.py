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
MODEL_PATH = ROOT / "models" / "finetuned_best.pt"
OUT = ROOT / "logs" / "sammenligning.json"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720
SIZE = 128
MARGIN = 0.4
SECONDS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
FINGER_NAMES = ["tommel", "peke", "lang", "ring", "lille"]

BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))


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

        return torch.stack([px, py], dim=2).view(b, 42)


model = HeatmapModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def crop_box(pts, w, h):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + MARGIN)
    return int(cx - side / 2), int(cy - side / 2), int(side)


def run_model(frame, left, top, side):
    h, w = frame.shape[:2]
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = frame[sy0:sy1, sx0:sx1]

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (SIZE, SIZE))
    x = torch.from_numpy(crop).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = (x - mean) / std

    with torch.no_grad():
        pred = model(x).view(21, 2).cpu().numpy()

    return pred * side + [left, top]


def draw(frame, pts, color, radius=4):
    pts = pts.astype(int)
    for a, b in BONES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
    for p in pts:
        cv2.circle(frame, tuple(p), radius, color, -1)


def run_test(name, instruction):
    print(f"\n=== {name} ===")
    print(instruction)
    print("Mellomrom = start, q = hopp over")

    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Mellomrom = start", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Sammenligning", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return None

    mine = []
    theirs = []
    mp_ms = []
    my_ms = []
    missed = 0
    total = 0

    start = time.perf_counter()
    while time.perf_counter() - start < SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        total += 1

        t0 = time.perf_counter()
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_ms.append((time.perf_counter() - t0) * 1000)

        if not result.multi_hand_landmarks:
            missed += 1
            cv2.putText(frame, "ingen hand", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Sammenligning", frame)
            cv2.waitKey(1)
            continue

        lm = result.multi_hand_landmarks[0]
        mp_pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

        left, top, side = crop_box(mp_pts, w, h)

        t0 = time.perf_counter()
        my_pts = run_model(frame, left, top, side)
        my_ms.append((time.perf_counter() - t0) * 1000)

        if my_pts is None:
            continue

        mine.append(my_pts)
        theirs.append(mp_pts)

        draw(frame, mp_pts, (255, 180, 0))
        draw(frame, my_pts, (0, 255, 0))
        cv2.rectangle(frame, (left, top), (left + side, top + side), (200, 200, 200), 1)

        igjen = SECONDS - (time.perf_counter() - start)
        cv2.putText(frame, f"{igjen:.1f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "blå = MediaPipe, grønn = egen", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Sammenligning", frame)
        cv2.waitKey(1)

    if len(mine) < 10:
        print("  for få bilder, hopper over")
        return None

    mine = np.array(mine)
    theirs = np.array(theirs)

    scale = np.linalg.norm(theirs[:, 0] - theirs[:, 9], axis=1).mean()

    diff = np.linalg.norm(mine - theirs, axis=2)

    jitter_mine = np.linalg.norm(np.diff(mine, axis=0), axis=2)
    jitter_theirs = np.linalg.norm(np.diff(theirs, axis=0), axis=2)

    result = {
        "test": name,
        "bilder": len(mine),
        "mediapipe_mistet_hand": missed,
        "mediapipe_ms": round(float(np.mean(mp_ms)), 1),
        "egen_modell_ms": round(float(np.mean(my_ms)), 1),
        "handstorrelse_px": round(float(scale), 1),
        "avvik_snitt_px": round(float(diff.mean()), 2),
        "avvik_snitt_andel_av_hand": round(float(diff.mean() / scale), 3),
        "avvik_verste_punkt_px": round(float(diff.mean(axis=0).max()), 2),
        "avvik_per_finger_px": {
            FINGER_NAMES[i]: round(float(diff[:, 1 + i * 4:5 + i * 4].mean()), 2)
            for i in range(5)
        },
        "skjelving_egen_px": round(float(jitter_mine.mean()), 2),
        "skjelving_mediapipe_px": round(float(jitter_theirs.mean()), 2),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


tests = [
    ("stein_rolig", "Hold knyttet neve stille"),
    ("papir_rolig", "Hold flat hånd stille"),
    ("saks_rolig", "Hold saks (peke + lang) stille"),
    ("saks_ring_lille", "Hold saks med ringfinger + lillefinger stille"),
    ("bevegelse", "Kast stein-saks-papir i normalt tempo"),
    ("avstand_naer", "Hold hånden naermere kameraet"),
    ("avstand_langt", "Hold hånden lenger unna"),
]

results = []
for name, instruction in tests:
    r = run_test(name, instruction)
    if r is not None:
        results.append(r)

cap.release()
cv2.destroyAllWindows()

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nLagret til {OUT}")
if results:
    print("\nOppsummering (avvik fra MediaPipe, i andel av handstørrelse):")
    for r in results:
        print(f"  {r['test']:18s} {r['avvik_snitt_andel_av_hand']:.3f}  "
              f"(skjelving: egen {r['skjelving_egen_px']:.2f}, mp {r['skjelving_mediapipe_px']:.2f})")
