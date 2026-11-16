import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.classification import gesture as G
from src.classification.testset import load_session, reference_rest

DATA = ROOT / "data" / "testset"
SESSIONS = ["magnus_test1", "magnus_test2"]
DETECTOR_PATH = ROOT / "models" / "detector_palm_neg_v2.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

TOLS = [0.05, 0.1]
FORMED_BY = ["straightness", "extension"]
SETTLE = 0.3
SEARCH = 1.0
NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}


def formed_time(r, tol, measure):
    times = np.array(r["times"])
    values = np.full((len(times), 4), np.nan)
    for k, p in enumerate(r["pts"]):
        if p is not None:
            if measure == "extension":
                values[k] = G.finger_extension(p)[1:]
            else:
                values[k] = G.finger_straightness(p)[1:]
    after = (times >= r["rest"]) & (times <= r["rest"] + SETTLE)
    if np.all(np.isnan(values[after])):
        return None
    final = np.nanmedian(values[after], axis=0)

    window = np.where((times >= r["rest"] - SEARCH) & (times <= r["rest"] + SETTLE))[0]
    formed = None
    for k in window[::-1]:
        if np.any(np.isnan(values[k])) or np.any(np.abs(values[k] - final) > tol):
            break
        formed = times[k]
    return formed


def lock_time(r):
    d = G.ThrowDecider()
    for p, t in zip(r["pts"], r["times"]):
        decision = d.update(p, t)
        if decision is not None:
            return decision, t
    return None, None


rounds = []
for s in SESSIONS:
    rounds += load_session(DATA / s, DETECTOR_PATH, LANDMARK_PATH)
for r in rounds:
    r["rest"] = reference_rest(r)
rounds = [r for r in rounds if r["rest"] is not None]

print("Regel:", "retthet" if G.USE_STRAIGHTNESS else "fingerstrekk",
      "terskel", G.STRAIGHT_THRESHOLD if G.USE_STRAIGHTNESS else G.THRESHOLD,
      " MARGIN", G.MARGIN, " FRAMES", G.FRAMES, " EARLY", G.EARLY)
print(f"{len(rounds)} runder fra {', '.join(SESSIONS)}")

for measure in FORMED_BY:
    for tol in TOLS:
        print(f"\n=== ferdig formet etter {measure}: alle fingre innenfor {tol} av sluttverdien ===")
        rows = {g: [] for g in NAMES}
        for r in rounds:
            decision, t = lock_time(r)
            formed = formed_time(r, tol, measure)
            if decision != r["label"]:
                continue
            if formed is None:
                formed = r["rest"]
            rows[r["label"]].append(((t - formed) * 1000, (t - r["rest"]) * 1000, (r["rest"] - formed) * 1000))

        everything = []
        for g in ["paper", "scissors"]:
            a = np.array(rows[g])
            if len(a) == 0:
                continue
            everything += rows[g]
            print(f"  {NAMES[g]:6s} {len(a):2d} kast   låst - ferdig: snitt {a[:, 0].mean():5.0f} ms, "
                  f"median {np.median(a[:, 0]):5.0f} ms, før ferdig {int((a[:, 0] < 0).sum())}/{len(a)}"
                  f"   |  fingrene ferdige før ro: median {np.median(a[:, 2]):4.0f} ms")
        a = np.array(everything)
        if len(a):
            print(f"  PAPIR + SAKS {len(a):2d} kast   låst - ferdig: snitt {a[:, 0].mean():5.0f} ms, "
                  f"median {np.median(a[:, 0]):5.0f} ms, før ferdig {int((a[:, 0] < 0).sum())}/{len(a)}")
        a = np.array(rows["rock"])
        if len(a):
            print(f"  STEIN  {len(a):2d} kast   låst - ro: snitt {a[:, 1].mean():5.0f} ms, "
                  f"median {np.median(a[:, 1]):5.0f} ms, før ro {int((a[:, 1] < 0).sum())}/{len(a)}")
print("\nBare riktige låsinger er med. 'ferdig før ro' = hvor lenge fingrene var ferdige før hånden stoppet.")