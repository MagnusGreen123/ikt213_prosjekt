import json
import pickle
import cv2
import numpy as np
from src.classification import gesture as G

REF_SPEED = 1.5
REF_TIME = 0.3

pipe = None


def load_session(folder, detector_path, landmark_path):
    global pipe
    cache = folder / f"landmarks_{detector_path.stem}_{landmark_path.stem}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    if pipe is None:
        from src.detection.pipeline import HandPipeline
        pipe = HandPipeline(detector_path, landmark_path, tracking=False)
        print("Enhet:", pipe.device)

    rounds = []
    round_dirs = sorted(folder.glob("round_*"))
    for n, rdir in enumerate(round_dirs):
        with open(rdir / "meta.json") as fh:
            meta = json.load(fh)
        pts_list, proc = [], []
        for i in range(len(meta["times"])):
            frame = cv2.imread(str(rdir / f"frame_{i:03d}.jpg"))
            if n == 0 and i == 0:
                for _ in range(10):
                    pipe.process(frame)
            pts, conf = pipe.process(frame)
            pts_list.append(None if pts is None else np.array(pts, dtype=float))
            proc.append(pipe.last.get("total_ms", 0))
        rounds.append({"name": f"{folder.name}/{rdir.name}", "label": meta["label"],
                       "times": meta["times"], "pts": pts_list, "proc_ms": proc})
        print(f"  {folder.name}/{rdir.name}: {len(pts_list)} bilder")

    with open(cache, "wb") as fh:
        pickle.dump(rounds, fh)
    print(f"Lagret {cache.name}")
    return rounds


def palm_y(r):
    palms = [G.palm_length(p) for p in r["pts"] if p is not None]
    palm = np.median(palms)
    return np.array([np.nan if p is None else p[G.PALM_POINTS, 1].mean() / palm for p in r["pts"]])


def reference_rest(r):
    pts, times = r["pts"], r["times"]
    if all(p is None for p in pts):
        return None
    y = palm_y(r)

    slow = [False] * len(pts)
    for k in range(1, len(pts)):
        if np.isnan(y[k]) or np.isnan(y[k - 1]):
            continue
        slow[k] = abs(y[k] - y[k - 1]) / (times[k] - times[k - 1]) < REF_SPEED

    rest = None
    k = 1
    while k < len(pts):
        if slow[k]:
            s = k
            while k < len(pts) and slow[k]:
                k += 1
            e = k - 1
            if times[e] - times[s - 1] >= REF_TIME and s - 1 > 0:
                rest = times[s - 1]
        else:
            k += 1
    return rest