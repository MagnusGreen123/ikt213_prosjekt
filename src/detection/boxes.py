import numpy as np

PALM_TO_BOX = 2.95


def palm_size(pts):
    return np.linalg.norm(pts[9] - pts[0])


def box_from_landmarks(pts):
    p = pts[:, :2] if pts.shape[1] > 2 else pts

    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    side = palm_size(p) * PALM_TO_BOX

    return cx - side / 2, cy - side / 2, side


def smooth_box(new_box, prev_box, blend=0.5, max_size_change=0.05):
    if prev_box is None:
        return new_box

    nl, nt, ns = new_box
    pl, pt, ps = prev_box

    ns = max(ps * (1 - max_size_change), min(ps * (1 + max_size_change), ns))

    ncx, ncy = nl + ns / 2, nt + ns / 2
    pcx, pcy = pl + ps / 2, pt + ps / 2
    cx = blend * pcx + (1 - blend) * ncx
    cy = blend * pcy + (1 - blend) * ncy

    return cx - ns / 2, cy - ns / 2, ns


def crop(frame, box, size, cv2):
    left, top, side = box
    left, top, side = int(left), int(top), int(side)
    if side < 10:
        return None

    h, w = frame.shape[:2]
    out = np.zeros((side, side, 3), dtype=frame.dtype)

    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    out[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (size, size))
