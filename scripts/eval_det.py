from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "detector_192_palm.npz"
MODEL_DIR = ROOT / "models"

BATCH = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Enhet:", device)


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


data = np.load(DATA)
images = torch.from_numpy(data["images"]).to(device)
boxes = torch.from_numpy(data["boxes"]).to(device)
throw_id = data["throw_id"]

n_throws = throw_id.max() + 1
val_throws = set(range(0, n_throws, 5))
is_val = np.array([t in val_throws for t in throw_id])
val_idx = torch.from_numpy(np.where(is_val)[0]).to(device)

print(f"{len(images)} bilder fra {n_throws} kast, {len(val_idx)} i validering")

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def evaluate(model):
    model.eval()
    ious, center_errors, size_errors, presences = [], [], [], []

    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH):
            idx = val_idx[start:start + BATCH]

            x = images[idx].permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            box = boxes[idx]

            pred_box, pred_pres = model(x)

            pcx, pcy, pside = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2].exp()
            tcx, tcy, tside = box[:, 0], box[:, 1], box[:, 2]

            center_errors.append(
                torch.sqrt((pcx - tcx) ** 2 + (pcy - tcy) ** 2).mean().item())
            size_errors.append((pside - tside).abs().mean().item())

            px0, py0 = pcx - pside / 2, pcy - pside / 2
            px1, py1 = pcx + pside / 2, pcy + pside / 2
            tx0, ty0 = tcx - tside / 2, tcy - tside / 2
            tx1, ty1 = tcx + tside / 2, tcy + tside / 2

            iw = (torch.min(px1, tx1) - torch.max(px0, tx0)).clamp(min=0)
            ih = (torch.min(py1, ty1) - torch.max(py0, ty0)).clamp(min=0)
            inter = iw * ih
            union = pside ** 2 + tside ** 2 - inter
            ious.append((inter / union.clamp(min=1e-9)).mean().item())

            presences.append((torch.sigmoid(pred_pres) >= 0.5).float().mean().item())

    return (float(np.mean(ious)), float(np.mean(center_errors)),
            float(np.mean(size_errors)), float(np.mean(presences)))


model = Detector().to(device)
results = []

for path in sorted(MODEL_DIR.glob("*.pt")):
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception:
        continue

    iou, center_err, size_err, presence = evaluate(model)
    results.append((path.name, iou, center_err, size_err, presence))
    print(f"  {path.name:28s} IoU {iou:.3f}   senterfeil {center_err * 100:.1f} %   "
          f"storrelsesfeil {size_err * 100:.1f} %   presence {presence:.3f}")

print(f"\n{'modell':28s} {'IoU':>7s} {'senter %':>9s} {'storrelse %':>12s} {'presence':>9s}")
for name, iou, c, s, p in results:
    print(f"{name:28s} {iou:7.3f} {c * 100:9.1f} {s * 100:12.1f} {p:9.3f}")