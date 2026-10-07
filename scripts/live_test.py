import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "finetuned_best.pt"
CAMERA_ID = 0
SIZE = 128
BOX = 300

device = "cuda" if torch.cuda.is_available() else "cpu"


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


BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


model = HeatmapModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

cap = cv2.VideoCapture(CAMERA_ID)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    x0 = w // 2 - BOX // 2
    y0 = h // 2 - BOX // 2
    crop = frame[y0:y0 + BOX, x0:x0 + BOX]

    start = time.perf_counter()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (SIZE, SIZE))
    x = torch.from_numpy(small).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = (x - mean) / std

    with torch.no_grad():
        pred = model(x).view(21, 2).cpu().numpy()
    ms = (time.perf_counter() - start) * 1000

    points = []
    for px, py in pred:
        points.append((int(x0 + px * BOX), int(y0 + py * BOX)))

    for i, (a, b) in enumerate(BONES):
        cv2.line(frame, points[a], points[b], COLORS[i // 4], 2)
    for p in points:
        cv2.circle(frame, p, 4, (255, 255, 255), -1)

    cv2.rectangle(frame, (x0, y0), (x0 + BOX, y0 + BOX), (200, 200, 200), 1)
    cv2.putText(frame, f"Modell: {ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
