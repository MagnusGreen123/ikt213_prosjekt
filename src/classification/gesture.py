import numpy as np

FINGER_BASE = [1, 5, 9, 13, 17]
PALM_POINTS = [0, 5, 9, 13, 17]

THRESHOLD = 0.5
MARGIN = 0.1

COUNT_TO_GESTURE = {0: "rock", 2: "scissors", 4: "paper"}

PUMPS = 2
FRAMES = 2
SHAPE_STILL = 0.1
AMPLITUDE = 0.3
BOUNCE = 0.6
REST_SPEED = 1.5
REST_FRAMES = 2
TIMEOUT = 1.5
MAX_MISSING = 5

USE_STRAIGHTNESS = True
STRAIGHT_THRESHOLD = 0.8

EARLY = True
LOOKAHEAD = 0.0
RISE_SPEED = 2.0
RISE_FOR_ALL = True
ROCK_FRAC = 0.1
MIN_STROKE = 3.0


def palm_length(pts):
    return np.linalg.norm(pts[9] - pts[0]) + 1e-9


def finger_extension(pts):
    scale = palm_length(pts)
    return np.array([np.linalg.norm(pts[b + 3] - pts[b]) / scale for b in FINGER_BASE])


def finger_straightness(pts):
    out = []
    for b in FINGER_BASE:
        chain = np.linalg.norm(pts[b + 1] - pts[b]) + np.linalg.norm(pts[b + 2] - pts[b + 1]) \
                + np.linalg.norm(pts[b + 3] - pts[b + 2])
        out.append(np.linalg.norm(pts[b + 3] - pts[b]) / (chain + 1e-9))
    return np.array(out)


def finger_values(pts):
    if USE_STRAIGHTNESS:
        return finger_straightness(pts)[1:], STRAIGHT_THRESHOLD
    return finger_extension(pts)[1:], THRESHOLD


def classify_frame(pts):
    ext, threshold = finger_values(pts)
    count = int((ext > threshold).sum())
    gesture = COUNT_TO_GESTURE.get(count)
    confidence = float(np.min(np.abs(ext - threshold)))
    return gesture, confidence


def classify_early(values, rise, threshold):
    ahead = values + rise * LOOKAHEAD
    rising = rise > RISE_SPEED
    extended = (values > threshold + MARGIN) | (rising & (ahead > threshold + MARGIN))
    bent = values < threshold - MARGIN
    if RISE_FOR_ALL:
        bent = bent & np.logical_not(rising)
    if not np.all(extended | bent):
        return None
    gesture = COUNT_TO_GESTURE.get(int(extended.sum()))
    if gesture == "rock" and np.any(rising):
        return None
    return gesture


class ThrowDecider:
    def __init__(self, pumps=PUMPS, frames=FRAMES, open_at=None):
        self.pumps_needed = pumps
        self.frames_needed = frames
        self.open_at = open_at
        self.reset()

    def reset(self):
        self.pumps = 0
        self.direction = None
        self.extreme = None
        self.y_min = None
        self.y_max = None
        self.bottoms = []
        self.bottom_y = None
        self.tops = []
        self.palm = None
        self.prev_y = None
        self.prev_t = None
        self.missing = 0
        self.slow_frames = 0
        self.window_open = False
        self.window_time = None
        self.window_tops = 0
        self.streak_gesture = None
        self.streak = 0
        self.prev_ext = None
        self.prev_values = None
        self.stroke_peak = 0.0
        self.tops_seen = 0
        self.last_still = False
        self.decision = None
        self.decision_time = None

    def _count_pumps(self, y, t):
        if self.direction is None:
            self.y_min = y if self.y_min is None else min(self.y_min, y)
            self.y_max = y if self.y_max is None else max(self.y_max, y)
            if y - self.y_min > AMPLITUDE:
                self.direction, self.extreme = "down", y
            elif self.y_max - y > AMPLITUDE:
                self.direction, self.extreme = "up", y
            return

        if self.direction == "down":
            if y > self.extreme:
                self.extreme = y
            elif self.extreme - y > AMPLITUDE:
                self.pumps += 1
                self.bottoms.append(t)
                self.bottom_y = self.extreme
                self.direction, self.extreme = "up", y
        else:
            if y < self.extreme:
                self.extreme = y
            elif y - self.extreme > AMPLITUDE:
                self.tops.append(t)
                self.direction, self.extreme = "down", y

    def _decide(self, gesture, t):
        self.decision = gesture
        self.decision_time = t
        return gesture

    def update(self, pts, t):
        if self.decision is not None:
            return self.decision

        if pts is None:
            self.missing += 1
            if self.missing >= MAX_MISSING:
                self.reset()
            return None
        self.missing = 0

        pl = palm_length(pts)
        self.palm = pl if self.palm is None else 0.8 * self.palm + 0.2 * pl

        y = pts[PALM_POINTS, 1].mean() / self.palm

        speed = 0.0
        dt = None
        if self.prev_y is not None and t > self.prev_t:
            dt = t - self.prev_t
            speed = abs(y - self.prev_y) / dt
        self.prev_y, self.prev_t = y, t

        values, threshold = finger_values(pts)
        rise = np.zeros(4)
        if self.prev_values is not None and dt is not None:
            rise = (values - self.prev_values) / dt
        self.prev_values = values

        ext = finger_extension(pts)[1:]
        still = self.prev_ext is not None and np.max(np.abs(ext - self.prev_ext)) < SHAPE_STILL
        self.prev_ext = ext
        self.last_still = bool(still)

        self._count_pumps(y, t)

        if len(self.tops) != self.tops_seen:
            self.tops_seen = len(self.tops)
            self.stroke_peak = 0.0
        if self.direction == "down":
            self.stroke_peak = max(self.stroke_peak, speed)

        if not self.window_open:
            if self.open_at is not None:
                opened = t >= self.open_at
            else:
                opened = self.pumps >= self.pumps_needed
            if not opened:
                return None
            self.window_open = True
            self.window_time = t
            self.window_tops = len(self.tops)

        if self.open_at is None and len(self.tops) > self.window_tops:
            self.window_time = t
            self.window_tops = len(self.tops)
            self.streak_gesture, self.streak = None, 0

        if t - self.window_time > TIMEOUT:
            self.reset()
            return None

        if EARLY:
            gesture = classify_early(values, rise, threshold)
            sure = gesture is not None
        else:
            gesture, confidence = classify_frame(pts)
            sure = gesture is not None and confidence >= MARGIN and still

        if sure:
            if gesture == self.streak_gesture:
                self.streak += 1
            else:
                self.streak_gesture, self.streak = gesture, 1
        else:
            self.streak_gesture, self.streak = None, 0

        self.slow_frames = self.slow_frames + 1 if speed < REST_SPEED else 0
        at_rest = self.slow_frames >= REST_FRAMES

        if self.streak_gesture in ("paper", "scissors") and self.streak >= self.frames_needed:
            return self._decide(self.streak_gesture, t)

        if EARLY and self.streak_gesture == "rock" and self.streak >= self.frames_needed \
                and self.direction == "down" and self.stroke_peak >= MIN_STROKE \
                and speed < ROCK_FRAC * self.stroke_peak:
            return self._decide("rock", t)

        at_bottom = self.direction == "down" or \
            (self.bottom_y is not None and self.bottom_y - y < BOUNCE)
        if at_rest and self.streak_gesture is not None and at_bottom:
            return self._decide(self.streak_gesture, t)

        return None