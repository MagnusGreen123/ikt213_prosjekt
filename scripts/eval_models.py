from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parent.parent
OWN = ROOT / "data" / "processed" / "own_128_palm.npz"
FREIHAND = ROOT / "data" / "processed" / "freihand_128.npz"
MODEL_DIR = ROOT / "models"

BATCH = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Enhet:", device)


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


mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def prepare(imgs, kps, idx):
    x = imgs[idx].permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    return x, kps[idx].reshape(-1, 42)


own = np.load(OWN, allow_pickle=True)
own_images = torch.from_numpy(own["images"]).to(device)
own_kp = torch.from_numpy(own["keypoints"]).to(device)
throw_id = own["throw_id"]

n_throws = throw_id.max() + 1
val_throws = set(range(0, n_throws, 5))
is_val = np.array([t in val_throws for t in throw_id])
own_val = torch.from_numpy(np.where(is_val)[0]).to(device)

print(f"Egne data: {len(own_images)} bilder fra {n_throws} kast, "
      f"{len(own_val)} i validering")


frei_images = frei_kp = frei_val = None
if FREIHAND.exists():
    frei = np.load(FREIHAND)
    frei_images = torch.from_numpy(frei["images"]).to(device)
    frei_kp = torch.from_numpy(frei["keypoints"]).to(device)
    pose_id = torch.from_numpy(frei["pose_id"]).to(device)
    frei_val = torch.where(pose_id >= 29000)[0]
    print(f"FreiHAND: {len(frei_val)} bilder i validering")
else:
    print("FreiHAND ikke funnet, hopper over den kolonnen")


def evaluate(model, images, keypoints, idx):
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(idx), BATCH):
            batch = idx[start:start + BATCH]
            x, y = prepare(images, keypoints, batch)
            pred = model(x)
            dist = (pred.float().view(-1, 21, 2) - y.view(-1, 21, 2)).norm(dim=2) * 224
            errors.append(dist.mean().item())
    return float(np.mean(errors))


model = HeatmapModel().to(device)
results = []

for path in sorted(MODEL_DIR.glob("*.pt")):
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        print(f"  hopper over {path.name} (annen arkitektur)")
        continue

    own_error = evaluate(model, own_images, own_kp, own_val)

    if frei_val is not None:
        frei_error = evaluate(model, frei_images, frei_kp, frei_val)
    else:
        frei_error = float("nan")

    results.append((path.name, own_error, frei_error))
    print(f"  {path.name:42s} egne {own_error:6.2f} px   freihand {frei_error:6.2f} px")


print(f"\n{'modell':42s} {'egne':>8s} {'freihand':>10s}")
for name, own_error, frei_error in results:
    print(f"{name:42s} {own_error:8.2f} {frei_error:10.2f}")

print("\nLaTeX-rader:")
for name, own_error, frei_error in results:
    safe = name.replace("_", r"\_")
    print(f"    \\texttt{{{safe}}} & {own_error:.2f} & {frei_error:.2f} \\\\")