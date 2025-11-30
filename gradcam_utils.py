# gradcam_utils.py
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model loader helpers
# -------------------------
def download_if_missing(local_path: str, url: str):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        import requests
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk)
        return local_path
    except Exception:
        if os.path.exists(local_path):
            return local_path
        raise

def load_baseline(path="models/cnn_baseline_fixed.pth", device=DEVICE):
    class CNNModel(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*16*16,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,num_classes)
            )
        def forward(self,x): return self.classifier(self.features(x))

    model = CNNModel(num_classes=4)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def load_resnet(path="models/resnet18_adapted.pth", device=DEVICE):
    # load torchvision resnet and adapt first conv -> 1 channel
    resnet = models.resnet18(weights=None)
    with torch.no_grad():
        w = resnet.conv1.weight.data
        new_w = w.mean(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(1, resnet.conv1.out_channels,
                                 kernel_size=resnet.conv1.kernel_size,
                                 stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=False)
        resnet.conv1.weight.copy_(new_w)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    state = torch.load(path, map_location=device)
    resnet.load_state_dict(state)
    resnet.to(device).eval()
    return resnet

# -------------------------
# Preprocessing helper
# -------------------------
_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def pil_to_tensor(pil_img):
    return _transform(pil_img.convert("L")).unsqueeze(0).to(DEVICE)

# -------------------------
# Basic Grad-CAM
# -------------------------
def find_last_conv(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def gradcam_for_image(model: nn.Module, pil_img: Image.Image, target_class: int | None = None):
    model.eval()
    tensor = pil_to_tensor(pil_img)  # (1,1,128,128)
    lastconv = find_last_conv(model)
    activations = {}
    gradients = {}

    def forward_hook(m, inp, out):
        activations['value'] = out.detach()

    def backward_hook(m, gin, gout):
        gradients['value'] = gout[0].detach()

    h1 = lastconv.register_forward_hook(forward_hook)
    h2 = lastconv.register_backward_hook(backward_hook)

    out = model(tensor)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    if target_class is None:
        target_class = int(out.argmax(dim=1).item())

    score = out[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=False)

    act = activations['value'].squeeze(0)    # (C,H,W)
    grad = gradients['value'].squeeze(0)     # (C,H,W)
    weights = grad.mean(dim=(1,2))           # (C,)

    cam = torch.relu((weights.view(-1,1,1) * act).sum(dim=0))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_np = cam.cpu().numpy()

    h1.remove(); h2.remove()
    return cam_np, target_class, probs
