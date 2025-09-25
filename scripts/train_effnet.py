import argparse, os
from pathlib import Path
import torch, timm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(data_dir, img_size=224, bs=32):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(Path(data_dir)/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_dir)/"val",   transform=val_tf)
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True))

def build_model():
    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

@torch.no_grad()
def evaluate(m, loader, device):
    m.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = m(x).squeeze(1)
        prob_fake = torch.sigmoid(logits)
        pred = (prob_fake > 0.5).long()  
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train(data_dir, out_path, epochs=5, lr=3e-4, bs=32, img_size=224, device=None):
    device = device or ("mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = get_loaders(data_dir, img_size, bs)
    m = build_model().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  

    best = 0.0
    for ep in range(1, epochs+1):
        m.train()
        for x, y in train_loader:
            x = x.to(device); y = y.float().to(device)
            logits = m(x).squeeze(1)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(m, val_loader, device)
        print(f"epoch {ep}: val_acc={acc:.3f}")
        if acc > best:
            best = acc
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(m.state_dict(), out_path)
            print("saved:", os.path.abspath(out_path))
    print("best_val_acc:", best)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/ and val/ subfolders")
    ap.add_argument("--out",  default="backend/models/weights/effnet_deepfake.pth")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.lr, args.bs, args.img_size)
