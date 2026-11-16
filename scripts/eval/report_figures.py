"""
Lagres i results/report/figures/
    early_curves_straight.png   fingrenes retthet over tid
    straightness_hist.png       retthet for bøyde og utstrakte fingre
    throw_timeline.pdf          én runde: håndens høyde, fingrene, vindu, låsing og ro
    dataset_examples.jpg        eksempler fra datasettene
    augment_examples.jpg        ett treningsutsnitt og augmenterte varianter

Slå av det du ikke trenger med MAKE_... = False.
"""

import sys
import json
import math
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "detection"))
from src.classification import gesture as G
from src.classification.testset import load_session, reference_rest, palm_y
from src.detection.boxes import box_from_landmarks

OUT = ROOT / "results" / "report" / "figures"
TEST_DIR = ROOT / "data" / "testset"
REC_DIR = ROOT / "data" / "recordings"
NEG_DIR = ROOT / "data" / "negatives"
FREI_DIR = ROOT / "data" / "raw" / "freihand"
OWN_NPZ = ROOT / "data" / "processed" / "own_128_palm_stor.npz"
SESSIONS = ["magnus_test1", "magnus_test2"]
DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

MAKE_CURVES = True
MAKE_HIST = True
MAKE_TIMELINE = True
MAKE_DATASET = True
MAKE_AUGMENT = True

HIST_LOG = True          # logaritmisk y-akse, så de få bøyde fingrene nær terskelen synes
TIMELINE_ROUND = None    # for eksempel "magnus_test2/round_020". None = første riktige papirrunde
SEED = 3                 # bytt for å få andre tilfeldige bilder i datasett-figuren

NAMES = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
FINGERS = ["index", "middle", "ring", "little"]
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
               (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]

OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
THR, MARGIN = G.STRAIGHT_THRESHOLD, G.MARGIN


def load_rounds():
    rounds = []
    for s in SESSIONS:
        if (TEST_DIR / s).exists():
            rounds += load_session(TEST_DIR / s, DETECTOR_PATH, LANDMARK_PATH)
    for r in rounds:
        r["rest"] = reference_rest(r)
    return [r for r in rounds if r["rest"] is not None]


def straight_values(r):
    t = (np.array(r["times"]) - r["rest"]) * 1000
    v = np.full((len(t), 4), np.nan)
    for k, p in enumerate(r["pts"]):
        if p is not None:
            v[k] = G.finger_straightness(p)[1:]
    return t, v


rounds = load_rounds() if (MAKE_CURVES or MAKE_HIST or MAKE_TIMELINE) else []

# =====================================================================
# Fingerkurver
# =====================================================================
if MAKE_CURVES:
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, g in zip(axes, NAMES):
        for r in rounds:
            if r["label"] != g:
                continue
            t, v = straight_values(r)
            keep = (t > -600) & (t < 300)
            for f in range(4):
                ax.plot(t[keep], v[keep, f], color=COLORS[f], lw=0.8, alpha=0.6)
        ax.axhspan(THR - MARGIN, THR + MARGIN, color="gold", alpha=0.25)
        ax.axhline(THR, color="black", lw=1)
        ax.axvline(0, color="red", lw=1, ls="--")
        ax.set_title(NAMES[g])
        ax.set_ylabel("straightness")
    for f in range(4):
        axes[0].plot([], [], color=COLORS[f], label=FINGERS[f])
    axes[0].legend(loc="lower left", fontsize=8, ncol=4)
    axes[-1].set_xlabel("ms relative to the moment the hand stops (red line)")
    plt.tight_layout()
    plt.savefig(OUT / "early_curves_straight.png", dpi=200)
    plt.close()
    print("Lagret early_curves_straight.png")

# =====================================================================
# Histogram
# =====================================================================
if MAKE_HIST:
    bent, extended = [], []
    for r in rounds:
        t, v = straight_values(r)
        for k in np.where((t >= 0) & (t <= 300))[0]:
            if np.any(np.isnan(v[k])):
                continue
            s = np.sort(v[k])
            if r["label"] == "rock":
                bent += list(s)
            elif r["label"] == "paper":
                extended += list(s)
            else:
                bent += list(s[:2])
                extended += list(s[2:])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, 1.0, 51)
    ax.hist(bent, bins=bins, alpha=0.75, label=f"bent ({len(bent)})")
    ax.hist(extended, bins=bins, alpha=0.75, label=f"extended ({len(extended)})")
    ax.axvspan(THR - MARGIN, THR + MARGIN, color="gold", alpha=0.3, label="threshold $\\pm$ margin")
    ax.axvline(THR, color="black", lw=1)
    if HIST_LOG:
        ax.set_yscale("log")
    ax.set_xlabel("straightness, 0 to 0.3 s after the hand stops")
    ax.set_ylabel("fingers" + (" (log scale)" if HIST_LOG else ""))
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "straightness_hist.png", dpi=200)
    plt.close()
    print("Lagret straightness_hist.png")

# =====================================================================
# Tidslinje for én runde
# =====================================================================
if MAKE_TIMELINE:
    chosen = None
    for r in rounds:
        if TIMELINE_ROUND is not None and r["name"] != TIMELINE_ROUND:
            continue
        if TIMELINE_ROUND is None and r["label"] != "paper":
            continue
        d = G.ThrowDecider()
        log = []
        lock_t = None
        for k, (p, t) in enumerate(zip(r["pts"], r["times"])):
            decision = d.update(p, t)
            log.append((t, d.pumps, d.window_open))
            if decision is not None:
                lock_t = t
                if decision == r["label"] or TIMELINE_ROUND is not None:
                    chosen = (r, log, lock_t, decision)
                break
        if chosen:
            break
    if chosen is None:
        print("Fant ingen runde til tidslinjen")
    else:
        r, log, lock_t, decision = chosen
        rest = r["rest"]
        t_all = (np.array(r["times"]) - rest) * 1000
        y = palm_y(r)
        height = np.nanmax(y) - y          # oppover i bildet = oppover i figuren
        t, v = straight_values(r)
        pump_times = [(tt - rest) * 1000 for i, (tt, pumps, w) in enumerate(log)
                      if i > 0 and pumps > log[i - 1][1]]
        window_times = [(tt - rest) * 1000 for i, (tt, pumps, w) in enumerate(log)
                        if w and (i == 0 or not log[i - 1][2])]
        lock_ms = (lock_t - rest) * 1000
        start = (window_times[0] if window_times else 0) - 900
        keep = (t_all > start) & (t_all < 400)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.2]})
        ax1.plot(t_all[keep], height[keep], color="black", lw=1.5)
        ax1.set_ylabel("hand height\n(palm lengths)")
        for f in range(4):
            ax2.plot(t[keep], v[keep, f], color=COLORS[f], lw=1.5, label=FINGERS[f])
        ax2.axhspan(THR - MARGIN, THR + MARGIN, color="gold", alpha=0.25)
        ax2.axhline(THR, color="black", lw=1)
        ax2.set_ylabel("straightness")
        ax2.set_xlabel("ms relative to the moment the hand stops")
        ax2.legend(fontsize=7, loc="best", ncol=2)
        for ax in (ax1, ax2):
            for i, pt in enumerate(pump_times):
                if pt > start:
                    ax.axvline(pt, color="0.5", ls=":", lw=1)
            if window_times:
                ax.axvspan(window_times[0], lock_ms, color="tab:green", alpha=0.08)
            ax.axvline(lock_ms, color="tab:green", lw=1.5)
            ax.axvline(0, color="red", ls="--", lw=1)
        top = ax1.get_ylim()[1]
        for i, pt in enumerate(pump_times):
            if pt > start:
                ax1.text(pt, top, f"pump {i + 1}", fontsize=7, ha="center", va="bottom", color="0.4")
        ax1.text(lock_ms, top * 0.97, f" locked {decision}", fontsize=7, ha="left", va="top", color="tab:green")
        ax1.text(0, top, "hand stops", fontsize=7, ha="center", va="bottom", color="red")
        if window_times:
            ax1.text(window_times[0], ax1.get_ylim()[0], " window open", fontsize=7, va="bottom", color="tab:green")
        plt.tight_layout()
        plt.savefig(OUT / "throw_timeline.pdf")
        plt.savefig(OUT / "throw_timeline.png", dpi=150)
        plt.close()
        print(f"Lagret throw_timeline.pdf ({r['name']}, låst {lock_ms:.0f} ms fra ro)")

# =====================================================================
# Eksempler fra datasettene
# =====================================================================
def draw_hand(img, pts, box=None):
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(int(c) for c in pts[a]), tuple(int(c) for c in pts[b]), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, tuple(int(c) for c in pts[a]), tuple(int(c) for c in pts[b]), (40, 40, 220), 2, cv2.LINE_AA)
    for x, yy in pts:
        cv2.circle(img, (int(x), int(yy)), 3, (40, 200, 255), -1, cv2.LINE_AA)
    if box is not None:
        left, top, side = box
        cv2.rectangle(img, (int(left), int(top)), (int(left + side), int(top + side)), (80, 220, 80), 2)


def crop_around(img, box, scale=1.6, size=260):
    left, top, side = box
    cx, cy, s = left + side / 2, top + side / 2, side * scale
    h, w = img.shape[:2]
    s = min(s, h, w)
    # flytt utsnittet inn i bildet i stedet for å fylle med svart
    x0 = int(min(max(cx - s / 2, 0), w - s))
    y0 = int(min(max(cy - s / 2, 0), h - s))
    pad = int(s)
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = big[y0 + pad:y0 + pad + int(s), x0 + pad:x0 + pad + int(s)]
    return cv2.resize(c, (size, size))


def tile_with_label(img, text, size=260):
    img = cv2.resize(img, (size, size)) if img.shape[:2] != (size, size) else img
    out = np.full((size + 30, size, 3), 255, np.uint8)
    out[:size] = img
    cv2.putText(out, text, (6, size + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def own_example(person, label, frame="last", variant=None):
    """Et bilde fra opptakene med punkter og boks, beskåret rundt hånden."""
    paths = []
    for path in sorted(REC_DIR.glob("*.npz")):
        d = np.load(path)
        if str(d["person"]) != person or str(d["label"]) != label:
            continue
        if variant is not None and ("variant" not in d.files or str(d["variant"]) != variant):
            continue
        paths.append(path)
    if not paths:
        return None
    path = random.choice(paths)
    d = np.load(path)
    lm = d["landmarks"]
    img_dir = REC_DIR / path.stem
    n = len(lm)
    if frame == "last":
        i = n - 1
    else:  # "blur": bildet der punktene flytter seg mest, midt i kastet
        moves = [np.nanmean(np.linalg.norm(lm[k, :, :2] - lm[k - 1, :, :2], axis=1)) for k in range(1, n)]
        i = int(np.nanargmax(moves)) + 1
    img = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
    if img is None:
        return None
    h, w = img.shape[:2]
    pts = lm[i, :, :2] * [w, h]
    box = box_from_landmarks(pts)
    draw_hand(img, pts, box)
    return crop_around(img, box)


def freihand_from_npz(n):
    """Leser n bilder fra data/processed/freihand_128.npz uten å laste hele filen.
    images-arrayen er flere GB, så vi hopper rett til bildene vi trenger."""
    import zipfile
    path = ROOT / "data" / "processed" / "freihand_128.npz"
    if not path.exists():
        return []
    print(f"  FreiHAND fra {path}")
    keypoints = np.load(path)["keypoints"]          # liten, delt på 224
    idx = sorted(random.sample(range(len(keypoints)), n))
    out = []
    with zipfile.ZipFile(path) as z:
        with z.open("images.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            start = f.tell()
            one = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
            for i in idx:
                f.seek(start + i * one)
                img = np.frombuffer(f.read(one), dtype=dtype).reshape(shape[1:])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_CUBIC)
                draw_hand(img, keypoints[i] * 260)
                out.append(img)
    return out


def freihand_examples(n):
    # 1. rådata (data/raw/freihand), 2. den ferdige freihand_128.npz
    if (FREI_DIR / "training_xyz.json").exists() and (FREI_DIR / "training" / "rgb").exists():
        print(f"  FreiHAND fra {FREI_DIR}")
        with open(FREI_DIR / "training_xyz.json") as f:
            xyz = json.load(f)
        with open(FREI_DIR / "training_K.json") as f:
            K = json.load(f)
        out = []
        for i in random.sample(range(len(xyz)), n):
            img = cv2.imread(str(FREI_DIR / "training" / "rgb" / f"{i:08d}.jpg"))
            p = np.array(xyz[i]) @ np.array(K[i]).T
            pts = p[:, :2] / p[:, 2:3]
            draw_hand(img, pts)
            out.append(img)
        return out
    out = freihand_from_npz(n)
    if not out:
        print("  FANT IKKE FreiHAND: verken data/raw/freihand eller data/processed/freihand_128.npz")
    return out


def negative_examples(n):
    files = sorted(NEG_DIR.glob("*/*.jpg"))
    if not files:
        return []
    out = []
    for f in random.sample(files, min(n, len(files))):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        side = min(h, w)
        out.append(img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2])
    return out


if MAKE_DATASET:
    tiles = []
    for img in freihand_examples(4):
        tiles.append(tile_with_label(img, "FreiHAND"))
    for person, label, frame, variant, text in [
        ("magnus", "rock", "last", None, "Own data, rock"),
        ("magnus", "paper", "last", None, "Own data, paper"),
        ("magnus", "scissors", "last", None, "Own data, scissors"),
        ("magnusrl", "scissors", "last", None, "Ring and little scissors"),
        ("magnus", "paper", "blur", None, "Mid-throw, blurred"),
        ("magnus", "scissors", "blur", None, "Mid-throw, blurred"),
        ("magnus2_opp", "paper", "last", None, "White wall, hand up"),
        ("Alan", "paper", "last", None, "Second person"),
    ]:
        img = own_example(person, label, frame, variant)
        if img is not None:
            tiles.append(tile_with_label(img, text))
        else:
            print(f"  fant ikke {person} {label}")
    for img in negative_examples(4):
        tiles.append(tile_with_label(img, "No hand (negative)"))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in tiles[i:i + cols]])
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(OUT / "dataset_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Lagret dataset_examples.jpg ({len(tiles)} ruter). Bytt SEED for andre bilder.")

# =====================================================================
# Augmentering
# =====================================================================
if MAKE_AUGMENT:
    import torch
    import torch.nn.functional as F
    from aug_extra import extra_augment

    def augment(x, y):
        """Samme geometri og lys som finetune.py."""
        b = x.shape[0]
        angle = torch.empty(b).uniform_(-90, 90) * math.pi / 180
        scale = torch.empty(b).uniform_(0.8, 1.15)
        shift = torch.empty(b, 2).uniform_(-0.10, 0.10)
        flip = torch.where(torch.rand(b) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
        A = torch.zeros(b, 2, 2)
        A[:, 0, 0] = cos * flip
        A[:, 0, 1] = -sin
        A[:, 1, 0] = sin * flip
        A[:, 1, 1] = cos
        p = torch.bmm(y * 2 - 1, A.transpose(1, 2)) + shift.unsqueeze(1)
        A_inv = torch.linalg.inv(A)
        theta = torch.zeros(b, 2, 3)
        theta[:, :, :2] = A_inv
        theta[:, :, 2] = -torch.bmm(A_inv, shift.unsqueeze(2)).squeeze(2)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        brightness = torch.empty(b, 1, 1, 1).uniform_(0.75, 1.25)
        x = extra_augment(x)
        contrast = torch.empty(b, 1, 1, 1).uniform_(0.75, 1.25)
        avg = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - avg) * contrast + avg) * brightness
        return x.clamp(0, 1), (p + 1) / 2

    torch.manual_seed(SEED)
    own = np.load(OWN_NPZ, allow_pickle=True)
    i = random.randrange(len(own["images"]))
    img = torch.from_numpy(own["images"][i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    kp = torch.from_numpy(own["keypoints"][i]).float().view(1, 21, 2)
    n = 7
    xs, ys = augment(img.repeat(n, 1, 1, 1), kp.repeat(n, 1, 1))
    tiles = []
    for k in range(n + 1):
        x = img[0] if k == 0 else xs[k - 1]
        y = kp[0] if k == 0 else ys[k - 1]
        im = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
        draw_hand(im, y.numpy() * 256)
        tiles.append(tile_with_label(im, "original" if k == 0 else f"augmented {k}", 256))
    rows = [np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in tiles[r:r + 4]])
            for r in range(0, len(tiles), 4)]
    cv2.imwrite(str(OUT / "augment_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Lagret augment_examples.jpg. Bytt SEED for et annet bilde og andre varianter.")

print(f"\nFigurene ligger i {OUT}. Kopier dem til figures/ i rapporten.")