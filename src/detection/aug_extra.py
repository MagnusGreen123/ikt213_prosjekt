import math
import torch
import torch.nn.functional as F

P_BLUR = 0.3
BLUR_SIZE = 9
P_GRAY = 0.2
SATURATION = (0.5, 1.5)
P_NOISE = 0.3
MAX_NOISE = 0.04
P_LOWRES = 0.2


def motion_blur(x):
    b, c, h, w = x.shape
    dev = x.device
    k = BLUR_SIZE

    r = torch.arange(k, device=dev) - k // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    theta = torch.rand(b, device=dev) * math.pi
    length = torch.empty(b, device=dev).uniform_(3, k)
    cos = torch.cos(theta).view(b, 1, 1)
    sin = torch.sin(theta).view(b, 1, 1)
    across = (xx * sin - yy * cos).abs()
    along = (xx * cos + yy * sin).abs()
    kernel = ((across < 0.5) & (along <= length.view(b, 1, 1) / 2)).float()
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)

    identity = torch.zeros(k, k, device=dev)
    identity[k // 2, k // 2] = 1.0
    use = (torch.rand(b, device=dev) < P_BLUR).view(b, 1, 1)
    kernel = torch.where(use, kernel, identity)

    weight = kernel.unsqueeze(1).expand(b, c, k, k).reshape(b * c, 1, k, k)
    out = F.conv2d(x.reshape(1, b * c, h, w), weight, padding=k // 2, groups=b * c)
    return out.view(b, c, h, w)


def color(x):
    b = x.shape[0]
    dev = x.device
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    sat = torch.empty(b, 1, 1, 1, device=dev).uniform_(*SATURATION)
    x = gray + (x - gray) * sat

    to_gray = torch.rand(b, 1, 1, 1, device=dev) < P_GRAY
    return torch.where(to_gray, gray.expand_as(x), x)


def noise_and_lowres(x):
    b, c, h, w = x.shape
    dev = x.device

    sigma = torch.empty(b, 1, 1, 1, device=dev).uniform_(0, MAX_NOISE)
    use = (torch.rand(b, 1, 1, 1, device=dev) < P_NOISE).float()
    x = x + torch.randn_like(x) * sigma * use

    small = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    small = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    use = torch.rand(b, 1, 1, 1, device=dev) < P_LOWRES
    return torch.where(use, small, x)


def extra_augment(x):
    if P_BLUR > 0:
        x = motion_blur(x)
    x = color(x)
    x = noise_and_lowres(x)
    return x.clamp(0, 1)