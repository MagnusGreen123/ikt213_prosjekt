

from pathlib import Path
import cv2
import numpy as np
from src.classification import gesture_2 as G

TILE = 400
BARS_H = 190
HEADER_H = 90
BG = (30, 24, 20)
WHITE = (240, 240, 240)
MUTED = (150, 140, 130)
GREEN = (110, 220, 90)
RED = (80, 80, 240)
YELLOW = (40, 200, 240)
FINGER_COLORS = [(40, 215, 255), (255, 170, 40), (110, 235, 90), (190, 70, 255), (40, 140, 255)]
FINGER_NAMES = ["index", "middle", "ring", "little"]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
FONT = cv2.FONT_HERSHEY_DUPLEX


def crop_square(frame, pts, scale=1.7):
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    side = int(max(x2 - x1, y2 - y1) * scale) + 20
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x0, y0 = cx - side // 2, cy - side // 2
    pad = side
    big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = big[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return crop, (x0, y0, side)


def draw_skeleton(img, pts):
    p = [(int(x), int(y)) for x, y in pts]
    for a, b in CONNECTIONS:
        col = MUTED if (a, b) in [(5, 9), (9, 13), (13, 17)] else FINGER_COLORS[(b - 1) // 4]
        cv2.line(img, p[a], p[b], (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(img, p[a], p[b], col, 3, cv2.LINE_AA)
    for i, pt in enumerate(p):
        cv2.circle(img, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 4, WHITE if i == 0 else FINGER_COLORS[(i - 1) // 4], -1, cv2.LINE_AA)


def draw_bars(img, ext, threshold, x, y, w):
    vmax = 1.2
    left = x + 80
    bar_w = w - 100

    def px(v):
        return left + int(min(v, vmax) / vmax * bar_w)

    top, bottom = y + 15, y + BARS_H - 35
    band = img[top:bottom, px(threshold - G.MARGIN):px(threshold + G.MARGIN)]
    band[:] = (band * 0.5 + np.array(YELLOW) * 0.15).astype(np.uint8)
    cv2.line(img, (px(threshold), top), (px(threshold), bottom), WHITE, 2)
    name = "straightness" if G.USE_STRAIGHTNESS else "extension"
    cv2.putText(img, f"{name}, threshold {threshold}", (px(threshold) - 110, y + BARS_H - 12), FONT, 0.5,
                MUTED, 1, cv2.LINE_AA)

    row_h = (bottom - top) // 4
    for i, (name, v) in enumerate(zip(FINGER_NAMES, ext)):
        yy = top + i * row_h + 6
        if v > threshold + G.MARGIN:
            col = GREEN
        elif v < threshold - G.MARGIN:
            col = RED
        else:
            col = YELLOW
        cv2.putText(img, name, (x + 8, yy + row_h // 2 + 2), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(img, (left, yy), (px(v), yy + row_h - 12), col, -1)
        cv2.putText(img, f"{v:.2f}", (min(px(v) + 6, x + w - 45), yy + row_h // 2 + 2), FONT, 0.45,
                    WHITE, 1, cv2.LINE_AA)


def info_text(info):
    if info is None:
        return "", MUTED
    if not info["window"]:
        return f"window closed, pumps {info['pumps']}", RED
    still = "still" if info["still"] else "moving"
    return f"window open, streak {info['streak']}, {still}", GREEN if info["still"] else YELLOW


def make_tile(frame, pts, label, info=None):
    tile = np.full((HEADER_H + TILE + BARS_H, TILE, 3), BG, np.uint8)
    line, col = info_text(info)
    cv2.putText(tile, line, (12, 75), FONT, 0.55, col, 1, cv2.LINE_AA)
    if pts is None:
        cv2.putText(tile, label, (12, 38), FONT, 0.8, WHITE, 1, cv2.LINE_AA)
        cv2.putText(tile, "no hand", (TILE // 2 - 60, HEADER_H + TILE // 2), FONT, 1.0, MUTED, 2, cv2.LINE_AA)
        return tile, None

    crop, (x0, y0, side) = crop_square(frame, pts)
    s = TILE / side
    crop = cv2.resize(crop, (TILE, TILE))
    draw_skeleton(crop, (pts - [x0, y0]) * s)
    tile[HEADER_H:HEADER_H + TILE] = crop

    g, c = G.classify_frame(pts)
    ext, threshold = G.finger_values(pts)
    sure = c >= G.MARGIN
    cv2.putText(tile, label, (12, 38), FONT, 0.8, WHITE, 1, cv2.LINE_AA)
    text = f"{(g or '-').upper()}  conf {c:.2f}"
    cv2.putText(tile, text, (TILE - 12 - cv2.getTextSize(text, FONT, 0.7, 1)[0][0], 38), FONT, 0.7,
                GREEN if (g and sure) else YELLOW, 1, cv2.LINE_AA)
    draw_bars(tile, ext, threshold, 0, HEADER_H + TILE, TILE)
    return tile, (x0, y0, side)


def save_lock_snapshot(out_dir, round_no, gesture, recent):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    lock_box = None
    n = len(recent)
    for i, item in enumerate(recent):
        frame, pts = item[0], item[1]
        info = item[2] if len(item) > 2 else None
        offset = i - (n - 1)
        label = "LOCK FRAME" if offset == 0 else f"frame {offset}"
        tile, box = make_tile(frame, pts, label, info)
        if offset == 0:
            lock_box = box
            cv2.rectangle(tile, (0, 0), (TILE - 1, tile.shape[0] - 1), GREEN, 3)
        tiles.append(tile)
    h = tiles[0].shape[0]

    frame = recent[-1][0].copy()
    if lock_box is not None:
        x0, y0, side = lock_box
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), GREEN, 4)
    fw = int(frame.shape[1] * (h - HEADER_H) / frame.shape[0])
    full = np.full((h, fw, 3), BG, np.uint8)
    full[HEADER_H:] = cv2.resize(frame, (fw, h - HEADER_H))
    cv2.putText(full, f"Round {round_no}: locked {gesture.upper()}", (12, 40), FONT, 0.9, WHITE, 1, cv2.LINE_AA)

    gap = np.full((h, 12, 3), BG, np.uint8)
    parts = [full]
    for t in tiles:
        parts += [gap, t]
    path = out_dir / f"lock_{round_no:03d}_{gesture}.jpg"
    cv2.imwrite(str(path), np.hstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path