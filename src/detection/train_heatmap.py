import math
import time
import subprocess
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_num_threads(6)
device = "cuda"

HOURS = 9          # hvor lenge treningen skal gå totalt
BATCH = 256
LR = 5e-4


# ---------- Ressursmåling ----------

def gpu_usage():
    # Returnerer (GPU-bruk i %, VRAM brukt i GB) fra nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split("\n")[0]
        util, mem = out.split(",")
        return float(util), float(mem) / 1024
    except Exception:
        return float("nan"), float("nan")


def ram_usage():
    # RAM brukt av denne prosessen i GB
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def container_ram():
    # RAM brukt av hele containeren i GB (grensen er 96 GB)
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read()) / 1024 ** 3
    except Exception:
        return float("nan")


# ---------- Modell (Simple Baselines + Integral Regression) ----------

class HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")
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


# ---------- Data ----------

data = np.load("data/processed/freihand_128.npz")
images = torch.from_numpy(data["images"]).to(device)
keypoints = torch.from_numpy(data["keypoints"]).to(device)
pose_id = torch.from_numpy(data["pose_id"]).to(device)

train_idx = torch.where(pose_id < 29000)[0]
val_idx = torch.where(pose_id >= 29000)[0]

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def augment(x, y):
    # x: (B, 3, H, W) i 0-1, y: (B, 21, 2) i 0-1
    b = x.shape[0]

    # Tilfeldig rotasjon, skalering, speiling og forskyvning
    angle = torch.empty(b, device=device).uniform_(-30, 30) * math.pi / 180
    scale = torch.empty(b, device=device).uniform_(0.85, 1.15)
    flip = torch.where(torch.rand(b, device=device) < 0.5, -1.0, 1.0)
    shift = torch.empty(b, 2, device=device).uniform_(-0.05, 0.05)

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale

    # A: flytter punkter fra originalbildet til det nye bildet (koordinater i -1 til 1)
    A = torch.zeros(b, 2, 2, device=device)
    A[:, 0, 0] = cos * flip
    A[:, 0, 1] = -sin
    A[:, 1, 0] = sin * flip
    A[:, 1, 1] = cos

    # Punktene
    p = y * 2 - 1
    p = torch.bmm(p, A.transpose(1, 2)) + shift.unsqueeze(1)
    y_new = (p + 1) / 2

    # Bildet: grid_sample trenger motsatt vei (nytt bilde -> original)
    A_inv = torch.linalg.inv(A)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, :, :2] = A_inv
    theta[:, :, 2] = -torch.bmm(A_inv, shift.unsqueeze(2)).squeeze(2)

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x_new = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    # Lysstyrke og kontrast
    brightness = torch.empty(b, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    contrast = torch.empty(b, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    avg = x_new.mean(dim=(1, 2, 3), keepdim=True)
    x_new = ((x_new - avg) * contrast + avg) * brightness

    return x_new.clamp(0, 1), y_new


def get_batch(idx, train):
    x = images[idx].permute(0, 3, 1, 2).float() / 255.0
    y = keypoints[idx]
    if train:
        x, y = augment(x, y)
    x = (x - mean) / std
    return x, y.reshape(-1, 42)


# ---------- Trening ----------

model = HeatmapModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")

best_error = float("inf")
total_epochs = None
scheduler = None
start_time = time.time()
epoch = 0

print(f"Startet {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

while total_epochs is None or epoch < total_epochs:
    epoch_start = time.time()
    cpu_start = time.process_time()

    model.train()
    order = train_idx[torch.randperm(len(train_idx), device=device)]
    train_losses = []
    gpu_samples = []

    for i, start in enumerate(range(0, len(order), BATCH)):
        x, y = get_batch(order[start:start + BATCH], train=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

        # Mål GPU-bruk jevnlig under treningen
        if i % 50 == 25:
            gpu_samples.append(gpu_usage())

    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH):
            x, y = get_batch(val_idx[start:start + BATCH], train=False)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            dist = (pred.float().view(-1, 21, 2) - y.view(-1, 21, 2)).norm(dim=2) * 224
            errors.append(dist.mean().item())

    val_error = np.mean(errors)
    epoch_time = time.time() - epoch_start
    lr = optimizer.param_groups[0]["lr"]

    # Etter første epoke: regn ut hvor mange epoker som får plass i tiden
    if total_epochs is None:
        total_epochs = max(2, int(HOURS * 3600 / epoch_time))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - 1)
        print(f"Epoke tar ca. {epoch_time:.0f} s, kjører {total_epochs} epoker totalt", flush=True)
    else:
        scheduler.step()

    # Ressurser for denne epoken
    cpu_percent = (time.process_time() - cpu_start) / epoch_time * 100   # 100 % = én kjerne
    gpu_util = np.mean([s[0] for s in gpu_samples]) if gpu_samples else float("nan")
    vram = max([s[1] for s in gpu_samples]) if gpu_samples else float("nan")

    # Tid
    elapsed = time.time() - start_time
    remaining = (total_epochs - epoch - 1) * epoch_time
    finish = datetime.now() + timedelta(seconds=remaining)

    print(f"[{datetime.now():%H:%M:%S}] Epoke {epoch + 1}/{total_epochs}: "
          f"train-loss {np.mean(train_losses):.5f}, val-feil {val_error:.2f} px, lr {lr:.6f}", flush=True)
    print(f"  Tid: {epoch_time:.0f} s/epoke, totalt {elapsed / 3600:.2f} t, "
          f"ferdig ca. {finish:%H:%M} (om {remaining / 3600:.1f} t)", flush=True)
    print(f"  GPU: {gpu_util:.0f} %, VRAM {vram:.1f} GB | CPU: {cpu_percent:.0f} % | "
          f"RAM: prosess {ram_usage():.1f} GB, container {container_ram():.1f} GB", flush=True)

    if val_error < best_error:
        best_error = val_error
        torch.save(model.state_dict(), "models/heatmap_best.pt")
        print(f"  Ny beste modell: {best_error:.2f} px", flush=True)

    torch.save(model.state_dict(), "models/heatmap_last.pt")
    epoch += 1

print(f"Ferdig. Beste val-feil: {best_error:.2f} px", flush=True)
