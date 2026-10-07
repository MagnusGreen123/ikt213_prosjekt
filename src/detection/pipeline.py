import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision

from .boxes import box_from_landmarks, smooth_box, crop

DET_SIZE = 192
LM_SIZE = 128
PRESENCE_THRESHOLD = 0.5


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        out = self.head(x)
        return out[:, :3], out[:, 3]


class HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        layers = []
        channels = 512
        for i in range(3):
            layers.append(nn.ConvTranspose2d(channels, 256, 4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            channels = 256
        self.upsample = nn.Sequential(*layers)
        self.head = nn.Conv2d(256, 21, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsample(x)
        heatmaps = self.head(x)

        b, k, h, w = heatmaps.shape
        prob = heatmaps.float().view(b, k, h * w).softmax(dim=2).view(b, k, h, w)

        xs = (torch.arange(w, device=x.device) + 0.5) / w
        ys = (torch.arange(h, device=x.device) + 0.5) / h
        px = (prob.sum(dim=2) * xs).sum(dim=2)
        py = (prob.sum(dim=3) * ys).sum(dim=2)

        confidence = prob.view(b, k, h * w).max(dim=2).values
        return torch.stack([px, py], dim=2).view(b, 42), confidence


class HandPipeline:
    def __init__(self, detector_path, landmark_path, device=None, tracking=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tracking = tracking

        self.detector = Detector().to(self.device)
        self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
        self.detector.eval()

        self.landmarker = HeatmapModel().to(self.device)
        self.landmarker.load_state_dict(torch.load(landmark_path, map_location=self.device))
        self.landmarker.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.prev_box = None

        self.last = {}

    def reset(self):
        self.prev_box = None

    def _to_tensor(self, img, size):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return (x - self.mean) / self.std

    def detect(self, frame):
        h, w = frame.shape[:2]
        x = self._to_tensor(frame, DET_SIZE)
        with torch.no_grad():
            box, presence = self.detector(x)
        cx = box[0, 0].item() * w
        cy = box[0, 1].item() * h
        side = box[0, 2].exp().item() * w
        p = torch.sigmoid(presence)[0].item()
        return (cx - side / 2, cy - side / 2, side), p

    def landmarks(self, frame, box):
        patch = crop(frame, box, LM_SIZE, cv2)
        if patch is None:
            return None, None

        x = self._to_tensor(patch, LM_SIZE)
        with torch.no_grad():
            pred, conf = self.landmarker(x)

        left, top, side = box
        pts = pred.view(21, 2).cpu().numpy() * side + [left, top]
        return pts, conf.view(21).cpu().numpy()

    def process(self, frame):
        h, w = frame.shape[:2]
        t_start = time.perf_counter()

        det_ms = None
        presence = 1.0
        used_detector = False

        if self.prev_box is None or not self.tracking:
            t0 = time.perf_counter()
            box, presence = self.detect(frame)
            det_ms = (time.perf_counter() - t0) * 1000
            used_detector = True
        else:
            box = self.prev_box

        pts = conf = None
        lm_ms = None
        collapsed = False

        if presence >= PRESENCE_THRESHOLD:
            t0 = time.perf_counter()
            pts, conf = self.landmarks(frame, box)
            lm_ms = (time.perf_counter() - t0) * 1000

        if pts is not None and self.tracking:
            new_box = box_from_landmarks(pts)
            if not used_detector:
                new_box = smooth_box(new_box, self.prev_box)

            nl, nt, ns = new_box
            if ns < 25 or nl + ns < 0 or nt + ns < 0 or nl > w or nt > h:
                collapsed = True
                self.prev_box = None
            else:
                self.prev_box = new_box
        elif pts is None:
            self.prev_box = None

        self.last = {
            "box": box,
            "presence": presence,
            "used_detector": used_detector,
            "collapsed": collapsed,
            "det_ms": det_ms,
            "lm_ms": lm_ms,
            "total_ms": (time.perf_counter() - t_start) * 1000,
        }
        return pts, conf


BONES = []
for base in [1, 5, 9, 13, 17]:
    BONES.append((0, base))
    BONES.append((base, base + 1))
    BONES.append((base + 1, base + 2))
    BONES.append((base + 2, base + 3))

FINGER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def draw_hand(frame, pts, color=None, dots=True):
    p = pts.astype(int)
    for i, (a, b) in enumerate(BONES):
        c = color if color is not None else FINGER_COLORS[i // 4]
        cv2.line(frame, tuple(p[a]), tuple(p[b]), c, 2)
    if dots:
        for q in p:
            cv2.circle(frame, tuple(q), 4, (255, 255, 255), -1)


def draw_box(frame, box, color):
    left, top, side = [int(v) for v in box]
    cv2.rectangle(frame, (left, top), (left + side, top + side), color, 2)
