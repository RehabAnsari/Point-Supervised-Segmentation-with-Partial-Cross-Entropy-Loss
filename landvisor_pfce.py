"""
Landvisor Project — Partial Cross-Entropy Loss for Point-Supervised Segmentation
=================================================================================

Task recap (from the assignment):
  1. Implement partial Cross Entropy loss (pfCE).
  2. Find/simulate a remote sensing segmentation dataset with point-level labels
     and plug the loss into a segmentation network.
  3. Design an experiment exploring a factor that affects performance, and
     report method / hypothesis / process / results.

Notes on this implementation
-----------------------------
- The loss is implemented exactly per the slide's formula:

        pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum(MASK_labeled)

  i.e. an ordinary (focal) cross-entropy loss, but averaged only over the
  pixels that actually have a point annotation instead of every pixel.

- This sandbox has no network access to real satellite-imagery datasets
  (e.g. ISPRS Potsdam, DeepGlobe, LoveDA), so a small synthetic remote-sensing-
  style land-cover dataset is generated procedurally (Perlin-like blobs make
  plausible "field/forest/water/urban" class regions). The dataset class has
  the same interface (image, full_mask) a real one would, so swapping in a
  real dataset just means replacing `SyntheticLandCoverDataset` with a class
  that reads real image/mask pairs — nothing else in the pipeline changes.

- Point supervision is simulated from the full ground-truth mask by randomly
  sampling N pixels per class per image and discarding the rest, which is
  exactly the annotation style described in the assignment ("points, or
  incomplete tagging").
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Partial (point-supervised) Focal Cross-Entropy loss
# ---------------------------------------------------------------------------
class PartialFocalCrossEntropy(nn.Module):
    """
    pfCE = sum_i( FocalLoss_i * mask_i ) / sum_i( mask_i )

    logits: (B, C, H, W) raw network outputs
    target: (B, H, W)    class index per pixel (ignored where mask == 0)
    mask:   (B, H, W)    1.0 where a point label exists, 0.0 elsewhere
    """

    def __init__(self, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Standard per-pixel CE (no reduction), then reweighted into a focal loss.
        # target has -1 (or anything) at unlabeled pixels, but that's fine since
        # the mask zeroes those contributions out before the sum.
        safe_target = target.clone()
        safe_target[mask == 0] = 0  # dummy class id so cross_entropy doesn't error on garbage labels

        ce = F.cross_entropy(logits, safe_target, reduction="none")  # (B, H, W)

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1).clamp(min=self.eps, max=1.0)
        focal_weight = (1.0 - pt) ** self.gamma

        focal_loss = focal_weight * ce  # (B, H, W)

        denom = mask.sum().clamp(min=1.0)
        pfce = (focal_loss * mask).sum() / denom
        return pfce


# ---------------------------------------------------------------------------
# 2. Point-label simulation
# ---------------------------------------------------------------------------
def simulate_point_labels(full_mask: np.ndarray, num_classes: int, points_per_class: int, rng: np.random.Generator):
    """
    Given a dense ground-truth mask (H, W) of class ids, keep only
    `points_per_class` randomly chosen pixels per class and discard the rest,
    mimicking sparse point annotation.

    Returns
    -------
    sparse_target : (H, W) int array, class id where labeled, 0 elsewhere (ignored via mask)
    label_mask    : (H, W) float array, 1.0 at labeled pixels, 0.0 elsewhere
    """
    H, W = full_mask.shape
    sparse_target = np.zeros((H, W), dtype=np.int64)
    label_mask = np.zeros((H, W), dtype=np.float32)

    for c in range(num_classes):
        ys, xs = np.where(full_mask == c)
        if len(ys) == 0:
            continue
        n = min(points_per_class, len(ys))
        chosen = rng.choice(len(ys), size=n, replace=False)
        sparse_target[ys[chosen], xs[chosen]] = c
        label_mask[ys[chosen], xs[chosen]] = 1.0

    return sparse_target, label_mask


# ---------------------------------------------------------------------------
# 3. Synthetic remote-sensing-style land-cover dataset
# ---------------------------------------------------------------------------
class SyntheticLandCoverDataset(Dataset):
    """
    Procedurally generates (image, mask) pairs that look like a coarse
    4-class land-cover map (0=water, 1=forest, 2=field, 3=urban), then
    simulates point annotations for each sample.

    Swap this class out for a real loader (ISPRS Potsdam, DeepGlobe, LoveDA,
    etc.) to run the exact same pipeline on real imagery — only __getitem__
    needs to return (image_tensor, full_mask) and the rest is unchanged.
    """

    NUM_CLASSES = 4

    def __init__(self, num_samples: int, size: int = 64, points_per_class: int = 20, seed: int = 0):
        self.num_samples = num_samples
        self.size = size
        self.points_per_class = points_per_class
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def _make_sample(self, rng: np.random.Generator):
        size = self.size
        # Blend a few random low-frequency sinusoids to get soft blob regions,
        # then threshold into NUM_CLASSES buckets -> a plausible land-cover map.
        yy, xx = np.mgrid[0:size, 0:size] / size
        field = np.zeros((size, size))
        for _ in range(4):
            fx, fy = rng.uniform(1, 4, size=2)
            phase = rng.uniform(0, 2 * np.pi)
            field += np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)

        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        mask = np.digitize(field, bins=[0.25, 0.5, 0.75]).astype(np.int64)  # values 0..3

        # Turn the mask into a fake 3-channel "satellite image" with class-dependent
        # colour statistics plus noise, so the network has something to learn from.
        palette = np.array([
            [30, 60, 200],   # water -> blue-ish
            [20, 120, 40],   # forest -> green
            [200, 180, 60],  # field -> yellow/tan
            [150, 150, 150], # urban -> gray
        ], dtype=np.float32)

        image = palette[mask]
        image += rng.normal(0, 15, size=image.shape)
        image = np.clip(image, 0, 255).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        return image, mask

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 100_000 + idx)
        image, full_mask = self._make_sample(rng)
        sparse_target, label_mask = simulate_point_labels(
            full_mask, self.NUM_CLASSES, self.points_per_class, rng
        )
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(sparse_target).long(),
            torch.from_numpy(label_mask).float(),
            torch.from_numpy(full_mask).long(),  # kept only for evaluation, never used in the loss
        )


# ---------------------------------------------------------------------------
# 4. A small U-Net-style segmentation network
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# 5. Training / evaluation helpers
# ---------------------------------------------------------------------------
def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Mean IoU computed against the FULL (dense) ground truth, for evaluation only."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        inter = (pred_c & target_c).sum().item()
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def run_training(points_per_class: int, epochs: int = 15, seed: int = 0, device: str = "cpu"):
    """Train TinyUNet with pfCE for a given point-annotation density and return final mIoU."""
    torch.manual_seed(seed)
    random.seed(seed)

    train_ds = SyntheticLandCoverDataset(num_samples=200, size=64, points_per_class=points_per_class, seed=seed)
    val_ds = SyntheticLandCoverDataset(num_samples=40, size=64, points_per_class=points_per_class, seed=seed + 999)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = TinyUNet(in_ch=3, num_classes=SyntheticLandCoverDataset.NUM_CLASSES).to(device)
    criterion = PartialFocalCrossEntropy(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for image, sparse_target, label_mask, _full_mask in train_loader:
            image, sparse_target, label_mask = image.to(device), sparse_target.to(device), label_mask.to(device)
            logits = model(image)
            loss = criterion(logits, sparse_target, label_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate against the FULL ground truth (only used for measuring performance,
    # never seen by the model during training).
    model.eval()
    ious = []
    with torch.no_grad():
        for image, _sparse_target, _label_mask, full_mask in val_loader:
            image, full_mask = image.to(device), full_mask.to(device)
            logits = model(image)
            pred = logits.argmax(dim=1)
            ious.append(mean_iou(pred, full_mask, SyntheticLandCoverDataset.NUM_CLASSES))

    return float(np.mean(ious))


# ---------------------------------------------------------------------------
# 6. Experiment: how does point-label density affect performance?
# ---------------------------------------------------------------------------
def run_experiment():
    """
    Factor under test: number of annotated points per class per image.
    Hypothesis: more points -> the partial CE loss sees a more representative
    sample of each class each step -> higher validation mIoU, with diminishing
    returns as density approaches "nearly dense" supervision.
    """
    point_densities = [2, 5, 10, 20, 50]
    results = {}

    for p in point_densities:
        miou = run_training(points_per_class=p, epochs=15, seed=42)
        results[p] = miou
        print(f"points_per_class={p:>3}  ->  val mIoU={miou:.4f}")

    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\nSummary:", results)
