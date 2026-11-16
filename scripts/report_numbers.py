import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.classification import gesture_2 as G

REC_DIR = ROOT / "data" / "recordings"
TEST_DIR = ROOT / "data" / "testset"
OUT_DIR = ROOT / "results" / "report"
SESSIONS = ["magnus_test1", "magnus_test2"]

DET_V1 = ROOT / "models" / "detector_palm_neg.pt"
DET_V2 = ROOT / "models" / "detector_palm_neg_v2.pt"
LM_OLD = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy.pt"
LM_FINAL = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy_magrl.pt"

LM_COMPARE = [
    ROOT / "models" / "finetuned_palm_frei_v3_70_stor.pt",
    LM_OLD,
    LM_FINAL,
]

RUN_DATASET = True
RUN_GESTURES = True
RUN_COMPARE = True
RUN_DECIDER = True
RUN_FORMED = True
RUN_HIST = True
RUN_TIMING = True
RUN_CI = True

END_FRAMES = 5
TIMING_FRAMES = 100
LIVE_COUNTS = [11, 34, 35, 74, 109]

NAMES = {"rock": "STEIN", "paper": "PAPIR", "scissors": "SAKS"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = open(OUT_DIR / "report_numbers.txt", "w", encoding="utf-8")


def say(text=""):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def compare(name, report_value, now_value):
    ok = "OK" if str(report_value) == str(now_value) else "AVVIK"
    say(f"  {name:55s} rapport {str(report_value):>10s}   nå {str(now_value):>10s}   {ok}")


def threshold():
    return G.STRAIGHT_THRESHOLD if G.USE_STRAIGHTNESS else G.THRESHOLD


def rule_text():
    return f"{'retthet' if G.USE_STRAIGHTNESS else 'fingerstrekk'} {threshold()}"


def measure_timing():
    from src.detection.pipeline import HandPipeline
    frames = sorted((TEST_DIR / SESSIONS[-1]).glob("round_*/frame_*.jpg"))[:TIMING_FRAMES]
    pipe = HandPipeline(DET_V2, LM_FINAL, tracking=False)
    images = [cv2.imread(str(f)) for f in frames]
    for img in images[:10]:
        pipe.process(img)
    ms = []
    for img in images:
        start = time.perf_counter()
        pipe.process(img)
        ms.append((time.perf_counter() - start) * 1000)
    ms = np.array(ms)
    print(f"  enhet {pipe.device}: median {np.median(ms):.1f} ms, snitt {ms.mean():.1f} ms, "
          f"maks {ms.max():.1f} ms per bilde, {1000 / np.median(ms):.0f} bilder/s ({len(ms)} bilder)")


if "--timing-only" in sys.argv:
    measure_timing()
    sys.exit()


say(f"Regel i gesture.py: {rule_text()}, MARGIN {G.MARGIN}, FRAMES {G.FRAMES}, EARLY {G.EARLY}, "
    f"RISE_FOR_ALL {getattr(G, 'RISE_FOR_ALL', '-')}, ROCK_FRAC {G.ROCK_FRAC}, BOUNCE {G.BOUNCE}")

if RUN_DATASET:
    say("\n=== 1. Datasettet per sesjon (tabell 3) ===")
    report_frames = {"magnus": 6826, "magnus2": 2138, "magnus2_opp": 2269, "magnus2_vesntre": 2177,
                     "magnusface": 3012, "magnusrl": 2328, "Alan": 2995, "Christian": 1068}
    rows = {}
    for path in sorted(REC_DIR.glob("*.npz")):
        data = np.load(path)
        person = str(data["person"])
        label = str(data["label"])
        if "variant" in data.files and label == "scissors" and str(data["variant"]) not in ("", "normal"):
            label = "scissors_" + str(data["variant"])
        n_images = len(list((REC_DIR / path.stem).glob("*.jpg")))
        r = rows.setdefault(person, {"throws": 0, "frames": 0, "images": 0, "labels": {}})
        r["throws"] += 1
        r["frames"] += len(data["landmarks"])
        r["images"] += n_images
        r["labels"][label] = r["labels"].get(label, 0) + 1
    total = 0
    for person, r in rows.items():
        labels = ", ".join(f"{k} {v}" for k, v in sorted(r["labels"].items()))
        say(f"  {person:18s} {r['throws']:4d} kast  {r['frames']:6d} bilder med punkter  "
            f"{r['images']:6d} bildefiler   ({labels})")
        total += r["frames"]
    say(f"  totalt {total} bilder med punkter")
    for person, n in report_frames.items():
        compare(f"bilder {person}", n, rows.get(person, {}).get("frames", "-"))
    neg = list((ROOT / "data").glob("**/*neg*"))
    say(f"  negative bilder: tell selv, mappene som ligner er: {[str(p.relative_to(ROOT)) for p in neg[:5]]}")

pipes = {}


def get_pipe(det, lm):
    from src.detection.pipeline import HandPipeline
    key = (det.name, lm.name)
    if key not in pipes:
        pipes[key] = HandPipeline(det, lm, tracking=False)
    return pipes[key]


def classify_values(values):
    if values is None:
        return "no hand"
    extended = int((values > threshold()).sum())
    return G.COUNT_TO_GESTURE.get(extended, "unsure")


def throw_values(npz_path, pipe):
    data = np.load(npz_path)
    img_dir = npz_path.parent / npz_path.stem
    n = len(data["landmarks"])
    ours, mp = [], []
    for i in range(n - 1, -1, -1):
        if len(ours) >= END_FRAMES:
            break
        frame = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        pts, conf = pipe.process(frame)
        if pts is not None:
            ours.append(G.finger_values(np.array(pts, dtype=float))[0])
        if len(mp) < END_FRAMES:
            mp_pts = data["landmarks"][i, :, :2] * np.array([w, h])
            if np.all(np.isfinite(mp_pts)):
                mp.append(G.finger_values(mp_pts)[0])
    ours_med = np.median(ours, axis=0) if len(ours) >= END_FRAMES else None
    mp_med = np.median(mp, axis=0) if mp else None
    return str(data["label"]), str(data["person"]), ours_med, mp_med


def evaluate(folder, person, det, lm):
    pipe = get_pipe(det, lm)
    rows = []
    for path in sorted(folder.glob("*.npz")):
        if str(np.load(path)["person"]) != person:
            continue
        rows.append(throw_values(path, pipe))
    result = {}
    for straight in (True, False):
        G.USE_STRAIGHTNESS = straight
        ours = sum(classify_values(o) == lab for lab, _, o, _ in rows)
        mp = sum(classify_values(m) == lab for lab, _, _, m in rows)
        result[straight] = (ours, mp, len(rows))
    G.USE_STRAIGHTNESS = USE_STRAIGHTNESS_ORIG
    return result


USE_STRAIGHTNESS_ORIG = G.USE_STRAIGHTNESS

if RUN_GESTURES:
    say("\n=== 2. Gestnøyaktighet per kast (tabell 6) ===")
    say("  Kast avgjøres av medianen over de siste 5 bildene, uten MARGIN, som i eval_gest_model.py.")
    cases = [
        ("Ring+lille holdt ute", TEST_DIR, "magnusrl", DET_V1, LM_OLD, "1/11", "11/11"),
        ("Ring+lille holdt ute", TEST_DIR, "magnusrl", DET_V1, LM_FINAL, "10/11", "11/11"),
        ("Ring+lille holdt ute", TEST_DIR, "magnusrl", DET_V2, LM_FINAL, "11/11", "11/11"),
        ("Ring+lille trening", REC_DIR, "magnusrl", DET_V1, LM_FINAL, "29/34", "-"),
        ("Ring+lille trening", REC_DIR, "magnusrl", DET_V2, LM_FINAL, "34/34", "34/34"),
        ("magnus2", REC_DIR, "magnus2", DET_V1, LM_FINAL, "32/33", "-"),
        ("magnus2", REC_DIR, "magnus2", DET_V2, LM_FINAL, "33/33", "33/33"),
    ]
    for name, folder, person, det, lm, rep_ours, rep_mp in cases:
        if not det.exists() or not lm.exists():
            say(f"  {name}: mangler {det.name if not det.exists() else lm.name}, hopper over")
            continue
        res = evaluate(folder, person, det, lm)
        say(f"\n  {name}, {det.name} + {lm.name}")
        for straight in (True, False):
            ours, mp, n = res[straight]
            rule = "retthet     " if straight else "fingerstrekk"
            say(f"    {rule}: oss {ours}/{n} ({ours / max(n, 1) * 100:.1f} %), MediaPipe {mp}/{n}")
        ours, mp, n = res[USE_STRAIGHTNESS_ORIG]
        compare(f"oss, {name}, {det.stem}", rep_ours, f"{ours}/{n}")
        if rep_mp != "-":
            compare(f"MediaPipe, {name}", rep_mp, f"{mp}/{n}")
    say("\n  Merk: 85.3 % = 29/34 og 97.0 % = 32/33. Ble de målt med fingerstrekk, sammenlign med den linjen.")

if RUN_COMPARE:
    say("\n=== 2b. Landemerkemodeller mot hverandre (for 80.8 / 97.0 og 57.6 / 100) ===")
    for lm in LM_COMPARE:
        if not lm.exists():
            say(f"  mangler {lm.name}")
            continue
        for det in (DET_V1, DET_V2):
            if not det.exists():
                continue
            for name, folder, person in [("magnus2", REC_DIR, "magnus2"), ("magnus (alle kast)", REC_DIR, "magnus")]:
                res = evaluate(folder, person, det, lm)
                parts = []
                for straight in (True, False):
                    ours, mp, n = res[straight]
                    parts.append(f"{'retthet' if straight else 'fingerstrekk'} {ours / max(n, 1) * 100:5.1f} %")
                say(f"  {lm.name:45s} {det.name:28s} {name:20s} " + "   ".join(parts))
    say("  Finn raden som gir 57.6 %, 80.8 % og 97.0 %, så vet du hvilket sett og hvilken detektor de gjelder.")
    say("  'magnus (alle kast)' inneholder treningskast, så det er ikke et rent testsett.")

if RUN_DECIDER or RUN_FORMED or RUN_HIST:
    from src.classification.testset import load_session, reference_rest
    rounds = []
    for s in SESSIONS:
        if (TEST_DIR / s).exists():
            rounds += load_session(TEST_DIR / s, DET_V2, LM_FINAL)
    for r in rounds:
        r["rest"] = reference_rest(r)
    rounds = [r for r in rounds if r["rest"] is not None]


def run_decider(r):
    d = G.ThrowDecider()
    for k, (p, t) in enumerate(zip(r["pts"], r["times"])):
        decision = d.update(p, t)
        if decision is not None:
            return decision, t, k
    return None, None, None


def decider_stats(subset, show_errors=True):
    lat_all, right, missing = [], 0, 0
    per = {g: [] for g in NAMES}
    for r in subset:
        decision, t, k = run_decider(r)
        if decision is None:
            missing += 1
            if show_errors:
                say(f"    {r['name']}: {NAMES[r['label']]} ble ikke låst")
            continue
        if decision != r["label"]:
            if show_errors:
                say(f"    {r['name']}: {NAMES[r['label']]} låst som {NAMES[decision]}")
            continue
        right += 1
        lat = (t - r["rest"]) * 1000 + r["proc_ms"][k]
        lat_all.append(lat)
        per[r["label"]].append(lat)
    lat_all = np.array(lat_all) if lat_all else np.array([np.nan])
    inside = int((lat_all <= 100).sum())
    return right, missing, inside, np.nanmedian(lat_all), np.nanmax(lat_all), per


def set_params(params):
    old = {k: getattr(G, k) for k in params}
    for k, v in params.items():
        setattr(G, k, v)
    return old


if RUN_DECIDER:
    say(f"\n=== 3. Decideren på {len(rounds)} testrunder ({', '.join(SESSIONS)}), tabell 8 ===")
    say(f"  Endelig oppsett fra gesture.py: {rule_text()}, EARLY {G.EARLY}, "
        f"RISE_FOR_ALL {getattr(G, 'RISE_FOR_ALL', '-')}, ROCK_FRAC {G.ROCK_FRAC}, BOUNCE {G.BOUNCE}")
    n = len(rounds)
    right, missing, inside, med, mx, per = decider_stats(rounds)
    say(f"  riktig {right}/{n}, ikke låst {missing}, innenfor 100 ms {inside}/{n}, "
        f"median {med:.0f} ms, maks {mx:.0f} ms")
    for g, v in per.items():
        if v:
            v = np.array(v)
            say(f"    {NAMES[g]:6s} median {np.median(v):5.0f} ms, maks {v.max():5.0f} ms, "
                f"før hånden stoppet {int((v < 0).sum())}/{len(v)}")
    compare("endelig: riktig", "60/60", f"{right}/{n}")
    compare("endelig: ikke låst", "0", f"{missing}")
    compare("endelig: innenfor 100 ms", "59/60", f"{inside}/{n}")
    compare("endelig: median (ms)", "-57", f"{med:.0f}")

    say("\n  Tidligere rader i tabell 8:")
    test1 = [r for r in rounds if r["name"].startswith("magnus_test1")]
    test2 = [r for r in rounds if r["name"].startswith("magnus_test2")]
    early = {"EARLY": True, "LOOKAHEAD": 0.0, "RISE_SPEED": 100.0, "ROCK_FRAC": 0.1, "BOUNCE": 0.6}
    rows = [
        ("Første decider, test2", test2, {"EARLY": False, "BOUNCE": 0.0}, ("39/48", "9", "-", "39")),
        ("+ sprett, test2", test2, {"EARLY": False, "BOUNCE": 0.6}, ("47/48", "1", "41/48", "40")),
        ("+ tidlig låsing, test2", test2, early, ("48/48", "0", "48/48", "-57")),
        ("samme, test1", test1, early, ("11/12", "0", "10/12", "-153")),
    ]
    for name, subset, params, (rep_right, rep_missing, rep_inside, rep_med) in rows:
        if not subset:
            say(f"  {name}: ingen runder")
            continue
        old = set_params(params)
        right, missing, inside, med, mx, per = decider_stats(subset, show_errors=False)
        set_params(old)
        m = len(subset)
        compare(f"{name}: riktig", rep_right, f"{right}/{m}")
        compare(f"{name}: ikke låst", rep_missing, f"{missing}")
        compare(f"{name}: innenfor 100 ms", rep_inside, f"{inside}/{m}")
        compare(f"{name}: median (ms)", rep_med, f"{med:.0f}")

    say("\n  Papir som ser ut som sikker saks (to fingre over terskel+MARGIN, to under terskel-MARGIN):")
    lo, hi = threshold() - G.MARGIN, threshold() + G.MARGIN
    for s in SESSIONS:
        papers = [r for r in rounds if r["label"] == "paper" and r["name"].startswith(s)]
        looks = 0
        for r in papers:
            t = np.array(r["times"]) - r["rest"]
            for k, p in enumerate(r["pts"]):
                if p is None or t[k] < -0.6 or t[k] > 0.3:
                    continue
                v = G.finger_values(p)[0]
                if (v > hi).sum() == 2 and (v < lo).sum() == 2:
                    looks += 1
                    break
        say(f"    {s}: {looks} av {len(papers)} papirrunder hadde et slikt bilde")
    say("  Rapporten sier 0 av 16 i magnus_test2 (der ble bare pekefinger+langfinger sjekket).")

def formed_time(r, tol, measure):
    times = np.array(r["times"])
    values = np.full((len(times), 4), np.nan)
    for k, p in enumerate(r["pts"]):
        if p is not None:
            if measure == "extension":
                values[k] = G.finger_extension(p)[1:]
            else:
                values[k] = G.finger_straightness(p)[1:]
    after = (times >= r["rest"]) & (times <= r["rest"] + 0.3)
    if np.all(np.isnan(values[after])):
        return None
    final = np.nanmedian(values[after], axis=0)
    window = np.where((times >= r["rest"] - 1.0) & (times <= r["rest"] + 0.3))[0]
    formed = None
    for k in window[::-1]:
        if np.any(np.isnan(values[k])) or np.any(np.abs(values[k] - final) > tol):
            break
        formed = times[k]
    return formed


if RUN_FORMED:
    say("\n=== 4. Låst mot ferdig formet (tabell 10) ===")
    report = {("extension", 0.05, "paper"): ("-56", "16/20"), ("extension", 0.05, "scissors"): ("32", "5/20"),
              ("extension", 0.1, "paper"): ("0", "9/20"), ("extension", 0.1, "scissors"): ("64", "2/20"),
              ("straightness", 0.1, "paper"): ("32", "3/20"), ("straightness", 0.1, "scissors"): ("63", "2/20")}
    locks = {r["name"]: run_decider(r) for r in rounds}
    for (measure, tol, g), (rep_med, rep_before) in report.items():
        diffs = []
        for r in rounds:
            decision, t, k = locks[r["name"]]
            if r["label"] != g or decision != g:
                continue
            f = formed_time(r, tol, measure)
            if f is not None:
                diffs.append((t - f) * 1000)
        diffs = np.array(diffs)
        if len(diffs) == 0:
            continue
        name = f"{measure} {tol} {NAMES[g]}"
        compare(f"{name}, median låst - ferdig (ms)", rep_med, f"{np.median(diffs):.0f}")
        compare(f"{name}, låst før ferdig", rep_before, f"{int((diffs < 0).sum())}/{len(diffs)}")

if RUN_HIST:
    say("\n=== 5. Retthet for bøyde og utstrakte fingre (figur 8) ===")
    bent, extended = [], []
    for r in rounds:
        t = np.array(r["times"]) - r["rest"]
        for k, p in enumerate(r["pts"]):
            if p is None or t[k] < 0 or t[k] > 0.3:
                continue
            v = np.sort(G.finger_straightness(p)[1:])
            if r["label"] == "rock":
                bent += list(v)
            elif r["label"] == "paper":
                extended += list(v)
            else:
                bent += list(v[:2])
                extended += list(v[2:])
    bent, extended = np.array(bent), np.array(extended)
    thr, m = G.STRAIGHT_THRESHOLD, G.MARGIN
    say(f"  bilder fra 0 til 0.3 s etter at hånden stoppet, {len(bent)} bøyde og {len(extended)} utstrakte fingre")
    say(f"  bøyd:     median {np.median(bent):.2f}, 99-persentil {np.percentile(bent, 99):.2f}, maks {bent.max():.2f}, "
        f"over {thr - m:.1f}: {np.mean(bent > thr - m) * 100:.1f} %")
    say(f"  utstrakt: median {np.median(extended):.2f}, 1-persentil {np.percentile(extended, 1):.2f}, "
        f"min {extended.min():.2f}, under {thr + m:.1f}: {np.mean(extended < thr + m) * 100:.1f} %")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        bins = np.linspace(0, 1.0, 51)
        plt.hist(bent, bins=bins, alpha=0.7, label="bent")
        plt.hist(extended, bins=bins, alpha=0.7, label="extended")
        plt.axvspan(thr - m, thr + m, color="gold", alpha=0.3, label="threshold $\\pm$ margin")
        plt.axvline(thr, color="black", lw=1)
        plt.xlabel("straightness")
        plt.ylabel("finger count")
        plt.legend()
        plt.tight_layout()
        path = OUT_DIR / "straightness_hist.png"
        plt.savefig(path, dpi=150)
        say(f"  lagret {path}")
    except ImportError:
        say("  matplotlib mangler, ingen figur (pip install matplotlib)")

if RUN_TIMING:
    say("\n=== 6. Behandlingstid per bilde (FR-14) ===")
    say("  Rapporten: 15.9 til 17.2 ms på RTX 4060 Laptop (First Release-diagnostikken).")
    for name, env in [("GPU", dict(os.environ)), ("CPU", {**os.environ, "CUDA_VISIBLE_DEVICES": ""})]:
        out = subprocess.run([sys.executable, __file__, "--timing-only"], env=env, capture_output=True,
                             text=True, cwd=ROOT)
        lines = [l for l in out.stdout.splitlines() if "enhet" in l]
        say(f"  {name}:" + (lines[0] if lines else f" feilet: {out.stderr.strip()[-300:]}"))
    say("  CPU-tallet gjelder bare modellene. For FR-14 må spillet også gå i 30 fps uten GPU.")

if RUN_CI:
    say("\n=== 7. Nedre grense i tosidig 95 % Clopper-Pearson-intervall for n av n riktige ===")
    for n in LIVE_COUNTS:
        say(f"  {n:4d} av {n:4d}: {0.025 ** (1 / n):.3f}")
    say("  Rapporten sier 0.90 for 35 av 35.")

say(f"\nAlt er lagret i {OUT_DIR / 'report_numbers.txt'}")
log_file.close()