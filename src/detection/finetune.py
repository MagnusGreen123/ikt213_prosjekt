import math
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parent.parent.parent
OWN = ROOT / "data" / "processed" / "own_128_palm.npz"
FREIHAND = ROOT / "data" / "processed" / "freihand_128.npz"
START_MODEL = ROOT / "models" / "heatmap_best.pt"
OUT_MODEL = ROOT / "models" / "finetuned_palm_frei_v3_70.pt"
EPOCHS = 60
BATCH = 128
LR = 1e-4
FREIHAND_SHARE = 0.7

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


own = np.load(OWN, allow_pickle=True)
own_images = torch.from_numpy(own["images"]).to(device)
own_kp = torch.from_numpy(own["keypoints"]).to(device)
throw_id = own["throw_id"]

print(f"Egne data: {len(own_images)} bilder fra {throw_id.max() + 1} kast")

n_throws = throw_id.max() + 1
val_throws = set(range(0, n_throws, 5))
is_val = np.array([t in val_throws for t in throw_id])

own_train = torch.from_numpy(np.where(~is_val)[0]).to(device)
own_val = torch.from_numpy(np.where(is_val)[0]).to(device)
print(f"  trening: {len(own_train)} bilder, validering: {len(own_val)} bilder")


frei_images = frei_kp = frei_train = None
if FREIHAND_SHARE > 0 and FREIHAND.exists():
    frei = np.load(FREIHAND)
    frei_images = torch.from_numpy(frei["images"]).to(device)
    frei_kp = torch.from_numpy(frei["keypoints"]).to(device)
    pose_id = torch.from_numpy(frei["pose_id"]).to(device)
    frei_train = torch.where(pose_id < 29000)[0]
    print(f"FreiHAND: {len(frei_train)} bilder blandes inn ({FREIHAND_SHARE:.0%} av hver batch)")
else:
    FREIHAND_SHARE = 0.0
    print("FreiHAND ikke funnet, trener bare på egne data")


mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def augment(x, y):
    b = x.shape[0]
    angle = torch.empty(b, device=device).uniform_(-25, 25) * math.pi / 180
    scale = torch.empty(b, device=device).uniform_(0.9, 1.1)
    flip = torch.where(torch.rand(b, device=device) < 0.5, -1.0, 1.0)
    shift = torch.empty(b, 2, device=device).uniform_(-0.04, 0.04)

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale

    A = torch.zeros(b, 2, 2, device=device)
    A[:, 0, 0] = cos * flip
    A[:, 0, 1] = -sin
    A[:, 1, 0] = sin * flip
    A[:, 1, 1] = cos

    p = y * 2 - 1
    p = torch.bmm(p, A.transpose(1, 2)) + shift.unsqueeze(1)
    y_new = (p + 1) / 2

    A_inv = torch.linalg.inv(A)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, :, :2] = A_inv
    theta[:, :, 2] = -torch.bmm(A_inv, shift.unsqueeze(2)).squeeze(2)

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x_new = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    brightness = torch.empty(b, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    contrast = torch.empty(b, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    avg = x_new.mean(dim=(1, 2, 3), keepdim=True)
    x_new = ((x_new - avg) * contrast + avg) * brightness

    return x_new.clamp(0, 1), y_new


def prepare(imgs, kps, idx, train):
    x = imgs[idx].permute(0, 3, 1, 2).float() / 255.0
    y = kps[idx]
    if train:
        x, y = augment(x, y)
    x = (x - mean) / std
    return x, y.reshape(-1, 42)


model = HeatmapModel().to(device)
model.load_state_dict(torch.load(START_MODEL, map_location=device))
print(f"Startet fra {START_MODEL.name}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

n_frei_per_batch = int(BATCH * FREIHAND_SHARE)
n_own_per_batch = BATCH - n_frei_per_batch


def validate():
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(own_val), BATCH):
            idx = own_val[start:start + BATCH]
            x, y = prepare(own_images, own_kp, idx, train=False)
            pred = model(x)
            dist = (pred.float().view(-1, 21, 2) - y.view(-1, 21, 2)).norm(dim=2) * 224
            errors.append(dist.mean().item())
    return float(np.mean(errors))


print(f"\nFør finjustering: val-feil {validate():.2f} px")
print(f"Starter {datetime.now():%H:%M:%S}\n")

best_error = float("inf")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()

    order = own_train[torch.randperm(len(own_train), device=device)]
    losses = []

    for start in range(0, len(order), n_own_per_batch):
        idx = order[start:start + n_own_per_batch]
        x, y = prepare(own_images, own_kp, idx, train=True)

        if n_frei_per_batch > 0:
            fidx = frei_train[torch.randint(len(frei_train), (n_frei_per_batch,), device=device)]
            fx, fy = prepare(frei_images, frei_kp, fidx, train=True)
            x = torch.cat([x, fx])
            y = torch.cat([y, fy])

        if scaler is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(model(x), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    val_error = validate()
    scheduler.step()

    print(f"[{datetime.now():%H:%M:%S}] Epoke {epoch + 1}/{EPOCHS}: "
          f"train-loss {np.mean(losses):.5f}, val-feil {val_error:.2f} px, "
          f"{time.time() - epoch_start:.0f} s", flush=True)

    if val_error < best_error:
        best_error = val_error
        torch.save(model.state_dict(), OUT_MODEL)
        print(f"  Ny beste modell: {best_error:.2f} px", flush=True)

print(f"\nFerdig på {(time.time() - start_time) / 60:.0f} min. Beste: {best_error:.2f} px")
print(f"Lagret til {OUT_MODEL}")
