import json
import numpy as np
import cv2

ROOT = "data/raw/freihand"
SIZE = 128
POSES = 32560  

with open(ROOT + "/training_xyz.json") as f:
    xyz = np.array(json.load(f))
with open(ROOT + "/training_K.json") as f:
    K = np.array(json.load(f))


keypoints = np.zeros((POSES, 21, 2), dtype=np.float32)
for i in range(POSES):
    uv = xyz[i] @ K[i].T
    keypoints[i] = uv[:, :2] / uv[:, 2:] / 224.0

total = POSES * 4
images = np.zeros((total, SIZE, SIZE, 3), dtype=np.uint8)
for i in range(total):
    img = cv2.imread(f"{ROOT}/training/rgb/{i:08d}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images[i] = cv2.resize(img, (SIZE, SIZE))
    if i % 10000 == 0:
        print(i, "/", total)

keypoints_all = np.concatenate([keypoints] * 4)
pose_id = np.arange(total) % POSES

np.savez("data/processed/freihand_128.npz",
         images=images, keypoints=keypoints_all, pose_id=pose_id)
print("Ferdig")
