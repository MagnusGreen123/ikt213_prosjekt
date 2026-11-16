import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "data" / "processed" / "detector_192_palm_v2.npz"
NEG_DATA = ROOT / "data" / "processed" / "negatives_192.npz"
OUT_MODEL = ROOT / "models" / "detector_palm_neg_v2.pt"

HOLDOUT = []
PERSONS = None

USE_EXTRA_AUG = True
EXTRA_BLUR_SIZE = 7

EPOCHS = 60
BATCH = 64
NEG_PER_BATCH = 16
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Enhet:", device)
if OUT_MODEL.exists():
    print(f"{OUT_MODEL.name} finnes allerede, endre OUT_MODEL så ingenting overskrives")
    raise SystemExit

extra_augment = None
if USE_EXTRA_AUG:
    import sys
    sys.path.insert(0, str(ROOT / "src" / "detection"))
    try:
        import aug_extra as augment_extra
        if hasattr(augment_extra, "BLUR_SIZE"):
            augment_extra.BLUR_SIZE = EXTRA_BLUR_SIZE
        extra_augment = augment_extra.extra_augment
        print("Ekstra augmentering: på")
    except ImportError:
        print("Fant ikke aug_extra.py i src/detection, kjører uten ekstra augmentering")


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
person_all = data["person"]
print("Personer i fila:", dict(zip(*np.unique(person_all, return_counts=True))))

for p in HOLDOUT:
    if not (person_all == p).any():
        print(f"Fant ingen bilder for HOLDOUT-personen '{p}', sjekk navnet")
        raise SystemExit
if PERSONS is None:
    PERSONS = [str(p) for p in np.unique(person_all) if p not in HOLDOUT]
for p in PERSONS:
    if not (person_all == p).any():
        print(f"Fant ingen bilder for '{p}', sjekk PERSONS")
        raise SystemExit
print("Trener på:", PERSONS)
print("Holdt ute:", HOLDOUT)

keep = np.isin(person_all, PERSONS + HOLDOUT)
images = torch.from_numpy(data["images"][keep])
boxes = torch.from_numpy(data["boxes"][keep])
throw_id = data["throw_id"][keep]
person = person_all[keep]

is_val = np.zeros(len(throw_id), dtype=bool)
for p in PERSONS:
    throws_p = np.unique(throw_id[person == p])
    is_val |= np.isin(throw_id, throws_p[::5])
is_holdout = np.isin(person, HOLDOUT)

train_idx = torch.from_numpy(np.where(~is_val & ~is_holdout)[0])
val_idx_per_person = {p: torch.from_numpy(np.where(is_val & (person == p))[0]) for p in PERSONS}
holdout_idx = {p: torch.from_numpy(np.where(person == p)[0]) for p in HOLDOUT}
print(f"Med hånd: {len(images)} bilder, trening {len(train_idx)}")
for p in PERSONS:
    print(f"  validering {p}: {len(val_idx_per_person[p])} bilder")
for p in HOLDOUT:
    print(f"  holdt ute {p}: {len(holdout_idx[p])} bilder")

neg = np.load(NEG_DATA)
neg_images = torch.from_numpy(neg["images"])
neg_group = neg["group"]
neg_is_val = neg_group % 5 == 0
neg_train_idx = torch.from_numpy(np.where(~neg_is_val)[0])
neg_val_idx = torch.from_numpy(np.where(neg_is_val)[0])
print(f"Uten hånd: {len(neg_images)} bilder, "
      f"trening {len(neg_train_idx)}, validering {len(neg_val_idx)}")

neg_weight = torch.ones(len(neg_images))

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

    x_new = x_new.clamp(0, 1)
    if extra_augment is not None:
        x_new = extra_augment(x_new)

    box_new = torch.stack([cx_new, cy_new, side_new.clamp(min=1e-3)], dim=1)
    return x_new.clamp(0, 1), box_new, presence


def prepare(idx, train):
    x = images[idx].to(device).permute(0, 3, 1, 2).float() / 255.0
    box = boxes[idx].to(device)
    if train:
        x, box, presence = augment(x, box)
    else:
        presence = torch.ones(len(idx), device=device)
    x = (x - mean) / std
    return x, box, presence


def prepare_negative(idx, train):
    x = neg_images[idx].to(device).permute(0, 3, 1, 2).float() / 255.0
    box = torch.tensor([0.5, 0.5, 0.2], device=device).repeat(len(idx), 1)
    if train:
        x, box, _ = augment(x, box)
    presence = torch.zeros(len(idx), device=device)
    x = (x - mean) / std
    return x, box, presence


def pick_negatives():
    weights = neg_weight[neg_train_idx]
    pick = torch.multinomial(weights, NEG_PER_BATCH, replacement=True)
    return neg_train_idx[pick]


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


def validate_hands(val_idx):
    ious = []
    center_errors = []
    size_errors = []
    hand_found = []
    for start in range(0, len(val_idx), BATCH):
        idx = val_idx[start:start + BATCH]
        x, box, presence = prepare(idx, train=False)
        pred_box, pred_pres = model(x)

        pcx, pcy, pside = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2].exp()
        tcx, tcy, tside = box[:, 0], box[:, 1], box[:, 2]

        center_errors.append(torch.sqrt((pcx - tcx) ** 2 + (pcy - tcy) ** 2))
        size_errors.append((pside - tside).abs())

        px0, py0 = pcx - pside / 2, pcy - pside / 2
        px1, py1 = pcx + pside / 2, pcy + pside / 2
        tx0, ty0 = tcx - tside / 2, tcy - tside / 2
        tx1, ty1 = tcx + tside / 2, tcy + tside / 2

        iw = (torch.min(px1, tx1) - torch.max(px0, tx0)).clamp(min=0)
        ih = (torch.min(py1, ty1) - torch.max(py0, ty0)).clamp(min=0)
        inter = iw * ih
        union = pside ** 2 + tside ** 2 - inter
        ious.append(inter / union.clamp(min=1e-9))

        hand_found.append((torch.sigmoid(pred_pres) >= 0.5).float())

    return (torch.cat(ious).mean().item(), torch.cat(center_errors).mean().item(),
            torch.cat(size_errors).mean().item(), torch.cat(hand_found).mean().item())


def validate():
    model.eval()
    results = {}
    neg_rejected = []
    with torch.no_grad():
        for p in PERSONS:
            results[p] = validate_hands(val_idx_per_person[p])
        for p in HOLDOUT:
            results[p] = validate_hands(holdout_idx[p])

        for start in range(0, len(neg_val_idx), BATCH):
            idx = neg_val_idx[start:start + BATCH]
            x, _, _ = prepare_negative(idx, train=False)
            _, pred_pres = model(x)
            neg_rejected.append((torch.sigmoid(pred_pres) < 0.5).float())

    neg_rate = torch.cat(neg_rejected).mean().item()
    return results, neg_rate


print(f"\nStarter {datetime.now():%H:%M:%S}\n")
best_score = 0.0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    order = train_idx[torch.randperm(len(train_idx))]
    losses = []

    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        x, box, presence = prepare(idx, train=True)

        neg_idx = pick_negatives()
        nx, nbox, npres = prepare_negative(neg_idx, train=True)

        x = torch.cat([x, nx])
        box = torch.cat([box, nbox])
        presence = torch.cat([presence, npres])

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

        neg_prob = torch.sigmoid(pred_pres[-len(neg_idx):].detach().float()).cpu()
        neg_weight[neg_idx] = neg_prob + 0.05

        losses.append(loss.item())

    results, neg_rate = validate()
    scheduler.step()

    print(f"[{datetime.now():%H:%M:%S}] Epoke {epoch + 1}/{EPOCHS}: "
          f"loss {np.mean(losses):.4f}, uten hånd avvist {neg_rate:.3f}, "
          f"{time.time() - epoch_start:.0f} s", flush=True)
    for p in PERSONS + HOLDOUT:
        iou, center_err, size_err, hand_rate = results[p]
        tag = "  (holdt ute)" if p in HOLDOUT else ""
        print(f"    {p:12s} IoU {iou:.3f}, senterfeil {center_err * 100:.1f}%, "
              f"størrelsesfeil {size_err * 100:.1f}%, hånd funnet {hand_rate:.3f}{tag}", flush=True)

    mean_iou = np.mean([results[p][0] for p in PERSONS])
    worst_hand = min(results[p][3] for p in PERSONS)
    score = mean_iou + worst_hand + neg_rate
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), OUT_MODEL)
        best_epoch = epoch + 1
        print(f"  Ny beste modell: snitt-IoU {mean_iou:.3f}, svakeste hånd funnet "
              f"{worst_hand:.3f}, uten hånd avvist {neg_rate:.3f}", flush=True)
        for p in HOLDOUT:
            print(f"    {p} (holdt ute) i denne modellen: IoU {results[p][0]:.3f}, "
                  f"hånd funnet {results[p][3]:.3f}", flush=True)

print(f"\nFerdig på {(time.time() - start_time) / 60:.0f} min. Beste epoke: {best_epoch}")
print(f"Lagret til {OUT_MODEL}")