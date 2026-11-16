

import sys
import itertools
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


GRID = {
    "MARGIN": [0.05, 0.1],
    "FRAMES": [1, 2],
    "EARLY": [False, True],
    "LOOKAHEAD": [0.0],
    "RISE_SPEED": [2.0, 5.0, 10.0, 100.0],
    "RISE_FOR_ALL": [True, False],
    "ROCK_FRAC": [0.0, 0.05, 0.1, 0.2],
    "BOUNCE": [0.6, 1.0],
}
DEFAULT = {"MARGIN": 0.1, "FRAMES": 2, "EARLY": True, "LOOKAHEAD": 0.0, "RISE_SPEED": 2.0,
           "RISE_FOR_ALL": True, "ROCK_FRAC": 0.1, "BOUNCE": 0.6}
EARLY_KEYS = ["LOOKAHEAD", "RISE_SPEED", "RISE_FOR_ALL", "ROCK_FRAC"]

LIMIT_MS = 100
SHOW_TOP = 15

NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}


def run_round(r, frames):
    d = G.ThrowDecider(frames=frames)
    for p, t, proc in zip(r["pts"], r["times"], r["proc_ms"]):
        decision = d.update(p, t)
        if decision is not None:
            return decision, t, proc
    return None, None, None


def evaluate(rounds, params):
    for k in params:
        if k != "FRAMES":
            setattr(G, k, params[k])
    right, none, lat, details = 0, 0, [], []
    for r in rounds:
        decision, t, proc = run_round(r, params["FRAMES"])
        if decision is None:
            none += 1
            details.append((r, None, None))
            continue
        ms = (t - r["rest"]) * 1000 + proc
        if decision == r["label"]:
            right += 1
            lat.append(ms)
        details.append((r, decision, ms))
    lat = np.array(lat)
    return {"params": params, "right": right, "none": none, "lat": lat, "details": details,
            "inside": int((lat <= LIMIT_MS).sum()) if len(lat) else 0,
            "median": float(np.median(lat)) if len(lat) else float("nan"),
            "max": float(lat.max()) if len(lat) else float("nan")}


def params_text(p):
    text = f"MARGIN {p['MARGIN']:<4} FRAMES {p['FRAMES']}  BOUNCE {p['BOUNCE']:<3}  EARLY {str(p['EARLY']):5}"
    if p["EARLY"]:
        text += (f"  LOOKAHEAD {p['LOOKAHEAD']:<5} RISE_SPEED {p['RISE_SPEED']:<5} "
                 f"RISE_FOR_ALL {str(p['RISE_FOR_ALL']):5} ROCK_FRAC {p['ROCK_FRAC']}")
    return text


def result_text(res, n):
    return (f"riktig {res['right']:3d}/{n}  ingen {res['none']:2d}  "
            f"innenfor {LIMIT_MS} ms {res['inside']:3d}/{n}  "
            f"median {res['median']:5.0f} ms  maks {res['max']:5.0f} ms")


def print_details(res):
    for g in NAMES:
        ms = [d[2] for d in res["details"] if d[0]["label"] == g and d[1] == g]
        if ms:
            print(f"    {NAMES[g]:6s} median {np.median(ms):5.0f} ms  maks {max(ms):5.0f} ms  "
                  f"innenfor {sum(m <= LIMIT_MS for m in ms)}/{len(ms)}")
    for r, decision, ms in res["details"]:
        if decision is None:
            print(f"    {r['name']}: {NAMES[r['label']]} ble ikke låst")
        elif decision != r["label"]:
            print(f"    {r['name']}: {NAMES[r['label']]} låst som {NAMES[decision]}")
        elif ms > LIMIT_MS:
            print(f"    {r['name']}: {NAMES[r['label']]} {ms:.0f} ms")


rounds = []
for s in SESSIONS:
    rounds += load_session(DATA / s, DETECTOR_PATH, LANDMARK_PATH)
print("Regel:", "retthet" if G.USE_STRAIGHTNESS else "fingerstrekk",
      "terskel", G.STRAIGHT_THRESHOLD if G.USE_STRAIGHTNESS else G.THRESHOLD)
usable = []
for r in rounds:
    r["rest"] = reference_rest(r)
    if r["rest"] is None:
        print(f"Hopper over {r['name']}: fant ingen tydelig ro etter kastet")
    else:
        usable.append(r)
n = len(usable)
print(f"\n{n} runder: " + ", ".join(f"{NAMES[g]} {sum(r['label'] == g for r in usable)}"
                                      for g in NAMES))

keys = list(GRID)
results = []
seen = set()
for values in itertools.product(*[GRID[k] for k in keys]):
    params = dict(zip(keys, values))
    if not params["EARLY"]:
        for k in EARLY_KEYS:
            params[k] = DEFAULT[k]
    key = tuple(params.values())
    if key in seen:
        continue
    seen.add(key)
    results.append(evaluate(usable, params))

results.sort(key=lambda res: (-res["right"], -res["inside"], res["median"]))

default = evaluate(usable, DEFAULT)
print(f"\nNå ({params_text(DEFAULT)})")
print(f"  {result_text(default, n)}")
print_details(default)

only_rock = dict(DEFAULT)
only_rock["RISE_FOR_ALL"] = False
res = evaluate(usable, only_rock)
print(f"\nFart bare for stein ({params_text(only_rock)})")
print(f"  {result_text(res, n)}")
print_details(res)

print(f"\nDe {SHOW_TOP} beste av {len(results)} kombinasjoner:")
for i, res in enumerate(results[:SHOW_TOP]):
    print(f"{i + 1:2d}. {params_text(res['params'])}")
    print(f"    {result_text(res, n)}")

print(f"\nDetaljer for nr. 1:")
print_details(results[0])

best_plain = [res for res in results if not res["params"]["EARLY"]][0]
best_early = [res for res in results if res["params"]["EARLY"]][0]
print("\nBeste uten EARLY:", params_text(best_plain["params"]))
print(f"  {result_text(best_plain, n)}")
print("Beste med EARLY: ", params_text(best_early["params"]))
print(f"  {result_text(best_early, n)}")
print_details(best_early)