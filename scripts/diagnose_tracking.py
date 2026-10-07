import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heatmap_best.pt"
CAMERA_ID = 0
SIZE = 128
BOX = 300
SECONDS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

FINGER_NAMES = ["tommel", "peke", "lang", "ring", "lille"]


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

        return torch.stack([px, py], dim=2), confidence


model = HeatmapModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

cap = cv2.VideoCapture(CAMERA_ID)


def predict(frame, x0, y0):
    crop = frame[y0:y0 + BOX, x0:x0 + BOX]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (SIZE, SIZE))
    x = torch.from_numpy(small).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = (x - mean) / std
    with torch.no_grad():
        points, conf = model(x)
    return points.view(21, 2).cpu().numpy(), conf.view(21).cpu().numpy()


def run_test(name, instruction):
    print(f"\n=== {name} ===")
    print(instruction)
    print("Trykk mellomrom for å starte, q for å hoppe over")

    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x0, y0 = w // 2 - BOX // 2, h // 2 - BOX // 2
        cv2.rectangle(frame, (x0, y0), (x0 + BOX, y0 + BOX), (200, 200, 200), 2)
        cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Mellomrom = start", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Diagnostikk", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return None

    all_points = []
    all_conf = []
    frame_times = []
    model_times = []
    start = time.perf_counter()

    while time.perf_counter() - start < SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        x0, y0 = w // 2 - BOX // 2, h // 2 - BOX // 2

        t0 = time.perf_counter()
        points, conf = predict(frame, x0, y0)
        model_times.append((time.perf_counter() - t0) * 1000)
        frame_times.append(time.perf_counter())

        all_points.append(points)
        all_conf.append(conf)

        for px, py in points:
            cv2.circle(frame, (int(x0 + px * BOX), int(y0 + py * BOX)), 4, (0, 255, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + BOX, y0 + BOX), (200, 200, 200), 2)
        igjen = SECONDS - (time.perf_counter() - start)
        cv2.putText(frame, f"{igjen:.1f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Diagnostikk", frame)
        cv2.waitKey(1)

    points = np.array(all_points) * BOX
    conf = np.array(all_conf)

    steps = np.linalg.norm(np.diff(points, axis=0), axis=2)

    fps = len(frame_times) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

    result = {
        "test": name,
        "bilder": len(points),
        "fps": round(fps, 1),
        "modell_ms_snitt": round(float(np.mean(model_times)), 1),
        "modell_ms_verste": round(float(np.max(model_times)), 1),
        "skjelving_snitt_px": round(float(steps.mean()), 2),
        "skjelving_verste_punkt_px": round(float(steps.mean(axis=0).max()), 2),
        "skjelving_per_finger": {
            FINGER_NAMES[i]: round(float(steps[:, 1 + i * 4:5 + i * 4].mean()), 2)
            for i in range(5)
        },
        "sikkerhet_snitt": round(float(conf.mean()), 4),
        "sikkerhet_laveste_punkt": round(float(conf.mean(axis=0).min()), 4),
        "sikkerhet_per_finger": {
            FINGER_NAMES[i]: round(float(conf[:, 1 + i * 4:5 + i * 4].mean()), 4)
            for i in range(5)
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


tests = [
    ("stein_rolig", "Hold en knyttet neve helt stille i firkanten"),
    ("papir_rolig", "Hold flat haand helt stille i firkanten"),
    ("saks_rolig", "Hold saks (peke + lang) helt stille"),
    ("saks_ring_lille", "Hold saks med ringfinger + lillefinger helt stille"),
    ("bevegelse", "Kast stein-saks-papir gjentatte ganger i normalt tempo"),
    ("avstand_naer", "Hold haanden naermere kameraet enn vanlig, rolig"),
    ("avstand_langt", "Hold haanden lenger unna kameraet, rolig"),
]

results = []
for name, instruction in tests:
    r = run_test(name, instruction)
    if r is not None:
        results.append(r)

cap.release()
cv2.destroyAllWindows()

out = ROOT / "logs" / "diagnostikk.json"
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nLagret til {out}")
