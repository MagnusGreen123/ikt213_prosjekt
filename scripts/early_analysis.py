
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.classification import gesture_2 as G
from src.classification.testset import load_session, reference_rest, palm_y

DATA = ROOT / "data" / "testset"
SESSIONS = ["magnus_test2"]
DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"
OUT = ROOT / "results" / f"early_curves_{'straight' if G.USE_STRAIGHTNESS else 'ext'}.png"

BEFORE = 0.6
AFTER = 0.3
FINGERS = ["index", "middle", "ring", "little"]
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}


def round_arrays(r):
    t = np.array(r["times"]) - r["rest"]
    n = len(t)
    values = np.full((n, 4), np.nan)
    for k, p in enumerate(r["pts"]):
        if p is not None:
            values[k], threshold = G.finger_values(p)
    rise = np.full((n, 4), np.nan)
    rise[1:] = (values[1:] - values[:-1]) / np.diff(t)[:, None]
    y = palm_y(r)
    speed = np.full(n, np.nan)
    speed[1:] = np.abs(np.diff(y)) / np.diff(t)
    return t, values, rise, speed, y


rounds = []
for s in SESSIONS:
    rounds += load_session(DATA / s, DETECTOR_PATH, LANDMARK_PATH)
rounds = [r for r in rounds if reference_rest(r) is not None]
for r in rounds:
    r["rest"] = reference_rest(r)

thr = G.STRAIGHT_THRESHOLD if G.USE_STRAIGHTNESS else G.THRESHOLD
lo, hi = thr - G.MARGIN, thr + G.MARGIN
print("Regel:", "retthet" if G.USE_STRAIGHTNESS else "fingerstrekk", "terskel", thr, "MARGIN", G.MARGIN)
print(f"{len(rounds)} runder: " + ", ".join(f"{NAMES[g]} {sum(r['label'] == g for r in rounds)}" for g in NAMES))

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for ax, g in zip(axes, NAMES):
    for r in rounds:
        if r["label"] != g:
            continue
        t, values, rise, speed, y = round_arrays(r)
        keep = (t > -BEFORE) & (t < AFTER)
        for f in range(4):
            ax.plot(t[keep] * 1000, values[keep, f], color=COLORS[f], lw=1, alpha=0.6)
    ax.axhspan(lo, hi, color="gold", alpha=0.25)
    ax.axhline(thr, color="black", lw=1)
    ax.axvline(0, color="red", lw=1, ls="--")
    ax.set_title(NAMES[g])
    ax.set_ylabel("straightness" if G.USE_STRAIGHTNESS else "extension")
for f in range(4):
    axes[0].plot([], [], color=COLORS[f], label=FINGERS[f])
axes[0].legend(loc="upper left", fontsize=8)
axes[-1].set_xlabel("ms fra hånden stoppet (rød linje)")
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=120)
print("Lagret", OUT)

print("\n1. PAPIR som ser ut som saks (pekefinger og langfinger ute, ring og lille inne)")
paper_worst = []
for r in rounds:
    if r["label"] != "paper":
        continue
    t, values, rise, speed, y = round_arrays(r)
    looks = (t > -BEFORE) & (values[:, 0] > hi) & (values[:, 1] > hi) & (values[:, 2] < lo) & (values[:, 3] < lo)
    if not looks.any():
        print(f"  {r['name']}: aldri")
        continue
    ring_little = np.max(rise[looks][:, 2:], axis=1)
    worst = float(np.min(ring_little))
    paper_worst.append(worst)
    times = ", ".join(f"{tt * 1000:.0f}" for tt in t[looks])
    print(f"  {r['name']}: {looks.sum()} bilder (ms {times}), "
          f"tregeste ring/lille steg {worst:.2f} per sekund")

print("\n2. SAKS: hvor fort ring og lille stiger i ekte saks")
scissors_max = []
for r in rounds:
    if r["label"] != "scissors":
        continue
    t, values, rise, speed, y = round_arrays(r)
    shape = (t > -BEFORE) & (t < AFTER) & (values[:, 0] > hi) & (values[:, 1] > hi) & (values[:, 2] < lo) & (values[:, 3] < lo)
    if not shape.any():
        print(f"  {r['name']}: fant ingen saks-bilder")
        continue
    m = float(np.nanmax(rise[shape][:, 2:]))
    scissors_max.append(m)
    print(f"  {r['name']}: raskeste ring/lille {m:.2f} per sekund")

if paper_worst and scissors_max:
    print(f"\n  Papir: tregeste {min(paper_worst):.2f}.  Saks: raskeste {max(scissors_max):.2f}.")
    if min(paper_worst) > max(scissors_max):
        print(f"  Skilles av farten. RISE_SPEED mellom {max(scissors_max):.2f} og {min(paper_worst):.2f}.")
    else:
        print("  Overlapper: i noen papir står ring og lille stille mens de andre er ute.")
        print("  Der kan ingen regel se forskjell, og EARLY vil enten låse feil eller vente.")

print("\n3. PAPIR og SAKS: siste knyttneve før fingrene åpner seg")
print("   (fart = håndens fart / toppfarten i nedslaget så langt)")
for r in rounds:
    if r["label"] == "rock":
        continue
    t, values, rise, speed, y = round_arrays(r)
    window = np.where((t > -BEFORE) & (t <= 0))[0]
    fist = [k for k in window if np.all(values[k] < lo)]
    if not fist:
        print(f"  {r['name']}: ingen knyttneve de siste {BEFORE * 1000:.0f} ms")
        continue
    k = fist[-1]
    start = window[0] + int(np.nanargmin(y[window[0]:k + 1]))
    peak = np.nanmax(speed[start:k + 1]) if k > start else np.nan
    print(f"  {r['name']}: {NAMES[r['label']]:6s} siste knyttneve {t[k] * 1000:5.0f} ms, "
          f"fart {speed[k] / peak if peak > 0 else np.nan:.2f} av toppfarten")
print("  Stein kan låses tidlig bare når ROCK_FRAC er under alle fart-tallene over,")
print("  og knyttneven ikke kommer etter tidspunktet stein ville blitt låst.")