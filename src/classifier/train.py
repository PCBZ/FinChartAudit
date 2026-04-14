"""Train a ViT-B chart misleader classifier on Misviz-synth 57K data.

Output: a 12-class multi-label classifier that scores each misleader type [0,1].
Used as verification layer for VLM predictions.

Usage:
    python train_classifier.py                # Train with defaults
    python train_classifier.py --epochs 5     # More epochs
    python train_classifier.py --eval-only    # Evaluate saved model on real-world
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

SYNTH_JSON = Path("data/misviz_synth/misviz_synth.json")
SYNTH_IMG_DIR = Path("data/misviz_synth/png")
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "chart_misleader_vit.pt"

MISLEADER_TYPES = [
    "misrepresentation",
    "3d",
    "truncated axis",
    "inappropriate use of pie chart",
    "inconsistent tick intervals",
    "dual axis",
    "inconsistent binning size",
    "discretized continuous variable",
    "inappropriate use of line chart",
    "inappropriate item order",
    "inverted axis",
    "inappropriate axis range",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(MISLEADER_TYPES)}
NUM_CLASSES = len(MISLEADER_TYPES)


class ChartDataset(Dataset):
    def __init__(self, items: list[dict], img_dir: Path, transform=None):
        self.items = items
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.img_dir / Path(item["image_path"]).name
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.zeros(NUM_CLASSES)
        for m in item.get("misleader", []):
            if m in TYPE_TO_IDX:
                label[TYPE_TO_IDX[m]] = 1.0
        return img, label


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.head = nn.Sequential(
        nn.Linear(model.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = json.loads(SYNTH_JSON.read_text(encoding="utf-8"))
    random.seed(42)
    random.shuffle(data)

    # Split: 90% train, 10% val
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ChartDataset(train_data, SYNTH_IMG_DIR, train_transform)
    val_ds = ChartDataset(val_data, SYNTH_IMG_DIR, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    print(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Freeze ViT backbone for first epoch, then unfreeze
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)

    # Class weights for imbalanced data
    pos_counts = torch.zeros(NUM_CLASSES)
    for item in train_data:
        for m in item.get("misleader", []):
            if m in TYPE_TO_IDX:
                pos_counts[TYPE_TO_IDX[m]] += 1
    neg_counts = len(train_data) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0
    for epoch in range(args.epochs):
        # Unfreeze backbone after epoch 0
        if epoch == 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            print("Backbone unfrozen")

        # Train
        model.train()
        train_loss = 0
        t0 = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] loss={loss.item():.4f}")

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Per-type metrics at threshold 0.5
        tp = ((all_preds > 0.5) & (all_labels == 1)).sum().item()
        fp = ((all_preds > 0.5) & (all_labels == 0)).sum().item()
        fn = ((all_preds <= 0.5) & (all_labels == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss/len(train_loader):.4f} "
              f"val_loss={val_loss/len(val_loader):.4f} "
              f"val_F1={f1:.3f} P={prec:.3f} R={rec:.3f} [{elapsed:.0f}s]")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "type_to_idx": TYPE_TO_IDX,
                "val_f1": f1,
                "epoch": epoch,
            }, MODEL_PATH)
            print(f"  Saved best model (F1={f1:.3f})")

    print(f"\nTraining done. Best val F1={best_val_f1:.3f}")
    print(f"Model saved to {MODEL_PATH}")

    # Per-type breakdown on best model
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(f"\nPer-type val metrics (threshold=0.5):")
    for i, t in enumerate(MISLEADER_TYPES):
        tp = ((all_preds[:, i] > 0.5) & (all_labels[:, i] == 1)).sum().item()
        fp = ((all_preds[:, i] > 0.5) & (all_labels[:, i] == 0)).sum().item()
        fn = ((all_preds[:, i] <= 0.5) & (all_labels[:, i] == 1)).sum().item()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        support = int(all_labels[:, i].sum())
        print(f"  {t:40s} P={p:.2f} R={r:.2f} F1={f:.2f} support={support}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    train(args)
