import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PATH = ROOT / "models" / "detector_best.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_best.pt"

CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 720

DET_SIZE = 192
LM_SIZE = 128
PRESENCE_THRESHOLD = 0.5

TRACKING = True
TRACK_MARGIN = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"

BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


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

        return torch.stack([px, py], dim=2).view(b, 42)


detector = Detector().to(device)
detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
detector.eval()

landmarker = HeatmapModel().to(device)
landmarker.load_state_dict(torch.load(LANDMARK_PATH, map_location=device))
landmarker.eval()

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


def box_from_landmarks(pts, w, h):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + TRACK_MARGIN)
    return int(cx - side / 2), int(cy - side / 2), int(side)


def find_landmarks(frame, left, top, side):
    h, w = frame.shape[:2]
    if side < 10:
        return None

    crop = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = frame[sy0:sy1, sx0:sx1]

    x = to_tensor(crop, LM_SIZE)
    with torch.no_grad():
        pred = landmarker(x).view(21, 2).cpu().numpy()

    return pred * side + [left, top]


cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

prev_box = None
det_times = []
lm_times = []
total_times = []
det_runs = 0
frames = 0

print("q = avslutt, d = skru sporing av/på")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frames += 1

    t_start = time.perf_counter()

    used_detector = False
    presence = 1.0

    if prev_box is None or not TRACKING:
        t0 = time.perf_counter()
        left, top, side, presence = detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)
        det_runs += 1
        used_detector = True
    else:
        left, top, side = prev_box

    pts = None
    if presence >= PRESENCE_THRESHOLD:
        t0 = time.perf_counter()
        pts = find_landmarks(frame, left, top, side)
        lm_times.append((time.perf_counter() - t0) * 1000)

    if pts is not None:
        prev_box = box_from_landmarks(pts, w, h)
        bl, bt, bs = prev_box
        if bs < 20 or bl + bs < 0 or bt + bs < 0 or bl > w or bt > h:
            prev_box = None
    else:
        prev_box = None

    total_times.append((time.perf_counter() - t_start) * 1000)

    if pts is not None:
        p = pts.astype(int)
        for i, (a, b) in enumerate(BONES):
            cv2.line(frame, tuple(p[a]), tuple(p[b]), COLORS[i // 4], 2)
        for q in p:
            cv2.circle(frame, tuple(q), 4, (255, 255, 255), -1)

    box_color = (0, 200, 255) if used_detector else (150, 150, 150)
    cv2.rectangle(frame, (left, top), (left + side, top + side), box_color, 2)

    lines = [
        f"Total: {np.mean(total_times[-30:]):.1f} ms",
        f"Detektor: {np.mean(det_times[-30:]):.1f} ms" if det_times else "Detektor: -",
        f"Landmarks: {np.mean(lm_times[-30:]):.1f} ms" if lm_times else "Landmarks: -",
        f"Detektor kjort: {det_runs}/{frames} bilder",
        f"Sporing: {'paa' if TRACKING else 'av'}",
    ]
    if presence < PRESENCE_THRESHOLD:
        lines.append(f"Ingen hand ({presence:.2f})")

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
        TRACKING = not TRACKING
        prev_box = None

cap.release()
cv2.destroyAllWindows()

print(f"\nBilder: {frames}")
print(f"Detektor kjørt: {det_runs} ({det_runs / max(frames, 1) * 100:.0f}% av bildene)")
if det_times:
    print(f"Detektor: {np.mean(det_times):.1f} ms i snitt")
if lm_times:
    print(f"Landmarks: {np.mean(lm_times):.1f} ms i snitt")
print(f"Totalt per bilde: {np.mean(total_times):.1f} ms i snitt, "
      f"{np.percentile(total_times, 95):.1f} ms i 95. persentil")
