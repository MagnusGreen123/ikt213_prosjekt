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
DATA = ROOT / "data" / "processed" / "detector_192_palm.npz"
OUT_MODEL = ROOT / "models" / "detector_palm.pt"

EPOCHS = 80
BATCH = 64
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Enhet:", device)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        out = self.head(x)
        box = out[:, :3]
        presence = out[:, 3]
        return box, presence


data = np.load(DATA)
images = torch.from_numpy(data["images"]).to(device)
boxes = torch.from_numpy(data["boxes"]).to(device)
throw_id = data["throw_id"]

print(f"{len(images)} bilder fra {throw_id.max() + 1} kast")

n_throws = throw_id.max() + 1
val_throws = set(range(0, n_throws, 5))
is_val = np.array([t in val_throws for t in throw_id])
train_idx = torch.from_numpy(np.where(~is_val)[0]).to(device)
val_idx = torch.from_numpy(np.where(is_val)[0]).to(device)
print(f"  trening: {len(train_idx)}, validering: {len(val_idx)}")

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def augment(x, box):
    b = x.shape[0]
    zoom = torch.empty(b, device=device).uniform_(0.6, 1.6)
    flip = torch.where(torch.rand(b, device=device) < 0.5, -1.0, 1.0)
    shift = torch.empty(b, 2, device=device).uniform_(-0.5, 0.5)

    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = zoom * flip
    theta[:, 1, 1] = zoom
    theta[:, 0, 2] = shift[:, 0]
    theta[:, 1, 2] = shift[:, 1]

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x_new = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    cx = box[:, 0] * 2 - 1
    cy = box[:, 1] * 2 - 1
    cx_new = (cx - shift[:, 0]) / (zoom * flip)
    cy_new = (cy - shift[:, 1]) / zoom
    side_new = box[:, 2] / zoom

    cx_new = (cx_new + 1) / 2
    cy_new = (cy_new + 1) / 2

    inside = (cx_new > 0.05) & (cx_new < 0.95) & (cy_new > 0.05) & (cy_new < 0.95)
    big_enough = (side_new > 0.03) & (side_new < 1.5)
    presence = (inside & big_enough).float()

    brightness = torch.empty(b, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    contrast = torch.empty(b, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    avg = x_new.mean(dim=(1, 2, 3), keepdim=True)
    x_new = ((x_new - avg) * contrast + avg) * brightness

    box_new = torch.stack([cx_new, cy_new, side_new.clamp(min=1e-3)], dim=1)
    return x_new.clamp(0, 1), box_new, presence


def prepare(idx, train):
    x = images[idx].permute(0, 3, 1, 2).float() / 255.0
    box = boxes[idx]
    if train:
        x, box, presence = augment(x, box)
    else:
        presence = torch.ones(len(idx), device=device)
    x = (x - mean) / std
    return x, box, presence


model = Detector().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None


def compute_loss(pred_box, pred_pres, box, presence):
    target = torch.stack([box[:, 0], box[:, 1], torch.log(box[:, 2])], dim=1)
    box_loss = F.smooth_l1_loss(pred_box, target, reduction="none").mean(dim=1)
    box_loss = (box_loss * presence).sum() / presence.sum().clamp(min=1)

    pres_loss = F.binary_cross_entropy_with_logits(pred_pres, presence)
    return box_loss + 0.1 * pres_loss, box_loss, pres_loss


def validate():
    model.eval()
    center_errors = []
    size_errors = []
    ious = []
    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH):
            idx = val_idx[start:start + BATCH]
            x, box, presence = prepare(idx, train=False)
            pred_box, _ = model(x)

            pcx, pcy, pside = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2].exp()
            tcx, tcy, tside = box[:, 0], box[:, 1], box[:, 2]

            center_errors.append(torch.sqrt((pcx - tcx) ** 2 + (pcy - tcy) ** 2).mean().item())
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

    return float(np.mean(ious)), float(np.mean(center_errors)), float(np.mean(size_errors))


print(f"\nStarter {datetime.now():%H:%M:%S}\n")
best_iou = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    order = train_idx[torch.randperm(len(train_idx), device=device)]
    losses = []

    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        x, box, presence = prepare(idx, train=True)

        if scaler is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                pred_box, pred_pres = model(x)
                loss, _, _ = compute_loss(pred_box.float(), pred_pres.float(), box, presence)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_box, pred_pres = model(x)
            loss, _, _ = compute_loss(pred_box, pred_pres, box, presence)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    iou, center_err, size_err = validate()
    scheduler.step()

    print(f"[{datetime.now():%H:%M:%S}] Epoke {epoch + 1}/{EPOCHS}: "
          f"loss {np.mean(losses):.4f}, IoU {iou:.3f}, "
          f"senterfeil {center_err * 100:.1f}% av bildet, "
          f"størrelsesfeil {size_err * 100:.1f}%, {time.time() - epoch_start:.0f} s", flush=True)

    if iou > best_iou:
        best_iou = iou
        torch.save(model.state_dict(), OUT_MODEL)
        print(f"  Ny beste modell: IoU {best_iou:.3f}", flush=True)

print(f"\nFerdig på {(time.time() - start_time) / 60:.0f} min. Beste IoU: {best_iou:.3f}")
print(f"Lagret til {OUT_MODEL}")
