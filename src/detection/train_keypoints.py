import numpy as np
import torch
import torchvision

device = "cuda"
EPOCHS = 50
BATCH = 256

data = np.load("data/processed/freihand_128.npz")
images = torch.from_numpy(data["images"]).to(device)
keypoints = torch.from_numpy(data["keypoints"]).to(device)
pose_id = torch.from_numpy(data["pose_id"]).to(device)

train_idx = torch.where(pose_id < 29000)[0]
val_idx = torch.where(pose_id >= 29000)[0]

model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(512, 42)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def prepare(idx):
    x = images[idx].permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    y = keypoints[idx].view(-1, 42)
    return x, y


for epoch in range(EPOCHS):
    model.train()
    order = train_idx[torch.randperm(len(train_idx), device=device)]

    for start in range(0, len(order), BATCH):
        x, y = prepare(order[start:start + BATCH])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH):
            x, y = prepare(val_idx[start:start + BATCH])
            pred = model(x).float().view(-1, 21, 2)
            dist = (pred - y.view(-1, 21, 2)).norm(dim=2) * 224
            errors.append(dist.mean().item())

    print(f"Epoke {epoch + 1}: loss {loss.item():.5f}, val-feil {np.mean(errors):.2f} px")
    torch.save(model.state_dict(), "models/keypoints_resnet18.pt")
