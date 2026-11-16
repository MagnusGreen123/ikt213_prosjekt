import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.classification import gesture as G

REC_DIR = ROOT / "data" / "recordings"
SOURCE = "model"
FRAMES_TO_TEST = [2, 3]
PERSONS = ["Christian"]
DETECTOR_PATH = ROOT / "models" / "detector_palm.pt"
LANDMARK_PATH = ROOT / "models" / "finetuned_palm_frei_v4_aug_heavy.pt"

GESTURES = ["rock", "paper", "scissors"]

pipe = None
if SOURCE == "model":
    from src.detection.pipeline import HandPipeline
    pipe = HandPipeline(DETECTOR_PATH, LANDMARK_PATH, tracking=False)
    print("Enhet:", pipe.device)


def load_throw(path):
    data = np.load(path)
    times = data["times"]
    img_dir = REC_DIR / path.stem
    first = cv2.imread(str(img_dir / "0000.jpg"))
    h, w = first.shape[:2] if first is not None else (720, 1280)

    if SOURCE == "mediapipe":
        frames = [lm[:, :2] * np.array([w, h]) for lm in data["landmarks"]]
    else:
        pipe.reset()
        frames = []
        for i in range(len(times)):
            img = cv2.imread(str(img_dir / f"{i:04d}.jpg"))
            pts = None
            if img is not None:
                pts, _ = pipe.process(img)
            frames.append(pts)

    return str(data["label"]), str(data["person"]), times, frames


def rest_time(times, frames):
    ys, ts = [], []
    palm = None
    for t, pts in zip(times, frames):
        if pts is None:
            continue
        pl = G.palm_length(pts)
        palm = pl if palm is None else 0.8 * palm + 0.2 * pl
        ys.append(pts[G.PALM_POINTS, 1].mean() / palm)
        ts.append(t)
    if not ts:
        return times[-1]
    rest = ts[-1]
    for i in range(len(ys) - 1, 0, -1):
        speed = abs(ys[i] - ys[i - 1]) / max(ts[i] - ts[i - 1], 1e-6)
        if speed >= G.REST_SPEED:
            break
        rest = ts[i - 1]
    return rest


def run(decider, times, frames):
    for t, pts in zip(times, frames):
        if decider.update(pts, t) is not None:
            break
    return decider


throws = []
for path in sorted(REC_DIR.glob("*.npz")):
    label, person, times, frames = load_throw(path)
    if PERSONS is not None and person not in PERSONS:
        continue

    missing = sum(f is None for f in frames) / max(len(frames), 1)
    if missing > 0.5:
        print(f"  {path.stem}: ingen hånd funnet i {missing * 100:.0f} % av bildene")

    probe = run(G.ThrowDecider(pumps=99), times, frames)
    t_rest = rest_time(times, frames)
    tops_before_rest = [t for t in probe.tops if t <= t_rest]
    last_top = tops_before_rest[-1] if tops_before_rest else times[0]

    throws.append((path.stem, label, times, frames, probe.pumps, last_top, t_rest, person))

print(f"{len(throws)} kast, landemerker fra {SOURCE}")
counts = np.bincount([t[4] for t in throws])
print("Pumper funnet per kast: " + ", ".join(f"{n}: {c}" for n, c in enumerate(counts) if c))
for person in sorted({t[7] for t in throws}):
    pumps = [t[4] for t in throws if t[7] == person]
    counts = np.bincount(pumps)
    print(f"  {person:16s} " + ", ".join(f"{n}: {c}" for n, c in enumerate(counts) if c))


def report(title, results):
    labels = np.array([r[0] for r in results])
    answers = np.array([r[1] for r in results])
    delays = np.array([r[2] for r in results], dtype=float)

    right = labels == answers
    none = answers == "none"
    print(f"\n{title}")
    print(f"  riktig {right.mean() * 100:.1f} %   feil {(~right & ~none).mean() * 100:.1f} %   "
          f"ingenting låst {none.mean() * 100:.1f} %")
    for g in GESTURES:
        ok = right & (labels == g)
        if ok.any():
            d = delays[ok] * 1000
            print(f"  {g:9s} låst {np.median(d):+6.0f} ms fra hånden stoppet (median), "
                  f"før stopp {np.mean(d < 0) * 100:3.0f} %, "
                  f"innen 100 ms etter {np.mean(d <= 100) * 100:3.0f} %")

    columns = GESTURES + ["none"]
    print(f"  {'fasit':10s} " + " ".join(f"{c:>9s}" for c in columns))
    for g in GESTURES:
        row = [np.sum((labels == g) & (answers == c)) for c in columns]
        print(f"  {g:10s} " + " ".join(f"{n:9d}" for n in row))


for frames_needed in FRAMES_TO_TEST:
    for mode in ["regel", "fasit"]:
        results = []
        for name, label, times, frames, pumps, last_top, t_rest, person in throws:
            if mode == "regel":
                d = G.ThrowDecider(frames=frames_needed)
            else:
                d = G.ThrowDecider(frames=frames_needed, open_at=last_top)
            run(d, times, frames)
            answer = d.decision if d.decision is not None else "none"
            delay = (d.decision_time - t_rest) if d.decision_time is not None else np.nan
            results.append((label, answer, delay))
        report(f"=== {frames_needed} bilder på rad, vindu fra {mode} ===", results)