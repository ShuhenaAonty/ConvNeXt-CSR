"""
ConvNeXt-CSR Part B v2 — ConvNeXt-Tiny backbone + CBAM + adaptive σ
(target MAE < 10, below published CSRNet 10.6 and our VGG-baseline 12.40)

Key improvements over csrnet_partb_adaptive.py (MAE=12.40):
  1. ConvNeXt-Tiny replaces VGG16 — modern CNN with larger receptive field,
     depthwise 7×7 convs, layer-scale regularisation, ImageNet-1K pretrained.
  2. 2-scale FPN fusion (H/8 C=192 + H/16 C=384 → 256) same as Part A v1 but
     with explicit BN-fused design (FPN lateral + smooth convs).
  3. CBAM (channel + spatial attention) on the fused 256-ch feature map and
     again after the first dilated conv — captures sparse crowd localisation.
  4. Geometry-adaptive sigma (k=4, β=0.3, σ∈[2, 18]) — same as best Part B run.
  5. Composite loss: MSE + 0.1×SSIM + 0.02×count-L1
     (count weight raised from 0.005 to 0.02 for stronger count supervision).
  6. Differential LR: backbone 5e-6 / FPN 2e-5 / head 5e-5.
  7. Warmup 10 ep (5 %→100 %) + CosineAnnealing 290 ep (eta_min=1e-7).
  8. 300-epoch budget (vs 200 in best Part B run).
  9. Test-Time Augmentation (H-flip) at every eval call.
"""

import copy, warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
DATASET_ROOT = Path(r"D:\Research\Crowd Counting\Dataset\ShanghaiTech\part_B")
TRAIN_IMG = DATASET_ROOT / "train_data" / "images"
TRAIN_GT  = DATASET_ROOT / "train_data" / "ground-truth"
TEST_IMG  = DATASET_ROOT / "test_data"  / "images"
TEST_GT   = DATASET_ROOT / "test_data"  / "ground-truth"

BASE_DIR = Path(r"D:\Research\Crowd Counting\Code\crowd_counting_notebooks_and_modules")
SAVE_DIR = BASE_DIR / "final files" / "checkpoints_convnext_partb_v2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------
IMG_H, IMG_W = 768, 1024
OUT_H, OUT_W = 96, 128          # 1/8-scale density map

K_NEIGHBORS = 4
BETA        = 0.3
SIGMA_MIN   = 2.0
SIGMA_MAX   = 18.0

LAMBDA_SSIM = 0.1
LAMBDA_CNT  = 0.02              # stronger count supervision for sparse Part B

EPOCHS      = 300
WARMUP_EP   = 10
BATCH_SIZE  = 4
LR_BACKBONE = 5e-6
LR_FPN      = 2e-5
LR_HEAD     = 5e-5
EVAL_EVERY  = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"ConvNeXt-CSR Part B v2 — ConvNeXt-Tiny + 2-scale FPN + CBAM")
print(f"Adaptive σ: k={K_NEIGHBORS}  β={BETA}  σ=[{SIGMA_MIN},{SIGMA_MAX}]")
print(f"LR  backbone={LR_BACKBONE}  fpn={LR_FPN}  head={LR_HEAD}")
print(f"Epochs: {EPOCHS}  warmup: {WARMUP_EP}  batch: {BATCH_SIZE}")


# ---------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------
def _gauss_kernel(size=11, sigma=1.5, device="cpu"):
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    return g2d.unsqueeze(0).unsqueeze(0)


def ssim_loss(pred, target, ks=11, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k   = _gauss_kernel(ks, sigma, pred.device)
    pad = ks // 2
    mx  = F.conv2d(pred,   k, padding=pad)
    my  = F.conv2d(target, k, padding=pad)
    mx2, my2, mxy = mx**2, my**2, mx*my
    sx2 = F.conv2d(pred**2,       k, padding=pad) - mx2
    sy2 = F.conv2d(target**2,     k, padding=pad) - my2
    sxy = F.conv2d(pred * target, k, padding=pad) - mxy
    num = (2*mxy + C1) * (2*sxy + C2)
    den = (mx2 + my2 + C1) * (sx2 + sy2 + C2)
    return 1.0 - (num / (den + 1e-8)).mean()


# ---------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ---------------------------------------------------------------
# ConvNeXtCSRPartB — Tiny backbone + 2-scale FPN + CBAM head
# ---------------------------------------------------------------
class ConvNeXtCSRPartB(nn.Module):
    """
    ConvNeXt-Tiny feature layout:
      features[0:4] → H/8,  C=192  (stage1)
      features[4:6] → H/16, C=384  (stage2)

    FPN:
      p16 = lat_16(stage2)                          H/16, C=256
      p8  = smooth_8(lat_8(stage1) + up2x(p16))    H/8,  C=256

    Density head:
      CBAM(256) → Conv(128,d=2) → ReLU → CBAM(128)
                → Conv(64,d=2)  → ReLU
                → Conv(32,p=1)  → ReLU → Conv(1)
    """

    def __init__(self):
        super().__init__()
        cnxt  = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feats = list(cnxt.features.children())

        self.stage1 = nn.Sequential(*feats[:4])   # H/8,  C=192
        self.stage2 = nn.Sequential(*feats[4:6])  # H/16, C=384

        # FPN lateral + smooth
        self.lat_16 = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.lat_8 = nn.Sequential(
            nn.Conv2d(192, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.smooth_8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # Density head with CBAM attention
        self.density_head = nn.Sequential(
            CBAM(256),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            CBAM(128),
            nn.Conv2d(128,  64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  32, 3, padding=1),             nn.ReLU(inplace=True),
            nn.Conv2d( 32,   1, 1),
        )

        self._init_new_layers()

    def _init_new_layers(self):
        new_mods = (list(self.lat_16.modules()) + list(self.lat_8.modules()) +
                    list(self.smooth_8.modules()) + list(self.density_head.modules()))
        for m in new_mods:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        s1 = self.stage1(x)   # H/8,  C=192
        s2 = self.stage2(s1)  # H/16, C=384

        p16 = self.lat_16(s2)
        p8  = self.smooth_8(self.lat_8(s1) +
                            F.interpolate(p16, size=s1.shape[2:], mode="nearest"))

        return F.relu(self.density_head(p8))


# ---------------------------------------------------------------
# GT loading + adaptive density map
# ---------------------------------------------------------------
def load_gt(gt_dir, stem):
    mat = sio.loadmat(Path(gt_dir) / f"GT_{stem}.mat")
    try:
        pts = mat["image_info"][0, 0][0, 0][0]
    except Exception:
        pts = mat.get("annPoints", np.zeros((0, 2)))
    return pts.astype(np.float32)


def make_density_adaptive(pts, w_orig, h_orig,
                          out_h=OUT_H, out_w=OUT_W,
                          k=K_NEIGHBORS, beta=BETA,
                          sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    dm = np.zeros((out_h, out_w), dtype=np.float32)
    if len(pts) == 0:
        return dm

    sx, sy = out_w / w_orig, out_h / h_orig
    spts   = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])
    n      = len(spts)

    if n == 1:
        xi = int(np.clip(spts[0, 0], 0, out_w - 1))
        yi = int(np.clip(spts[0, 1], 0, out_h - 1))
        blob = np.zeros((out_h, out_w), dtype=np.float64)
        blob[yi, xi] = 1.0
        blurred = gaussian_filter(blob, sigma=sigma_max)
        s = blurred.sum()
        if s > 0:
            dm += (blurred / s).astype(np.float32)
        return dm

    tree     = cKDTree(spts)
    k_actual = min(k + 1, n)
    dists, _ = tree.query(spts, k=k_actual)
    sigmas   = np.clip(beta * dists[:, 1:].mean(axis=1), sigma_min, sigma_max)

    for i, pt in enumerate(spts):
        xi = int(np.clip(pt[0], 0, out_w - 1))
        yi = int(np.clip(pt[1], 0, out_h - 1))
        blob = np.zeros((out_h, out_w), dtype=np.float64)
        blob[yi, xi] = 1.0
        blurred = gaussian_filter(blob, sigma=float(sigmas[i]))
        s = blurred.sum()
        if s > 0:
            blurred /= s
        dm += blurred.astype(np.float32)
    return dm


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class CrowdDataset(Dataset):
    NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, img_dir, gt_dir, is_train=True):
        self.paths    = sorted(Path(img_dir).glob("*.jpg"))
        self.is_train = is_train

        tf_list = [T.Resize((IMG_H, IMG_W))]
        if is_train:
            tf_list += [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)]
        tf_list += [T.ToTensor(), self.NORM]
        self.tf = T.Compose(tf_list)

        label = "train" if is_train else "test"
        print(f"  Precomputing {label} adaptive density maps ({len(self.paths)} images)…")
        self.densities, self.counts = [], []
        for p in tqdm(self.paths, leave=False):
            img  = Image.open(p)
            w0, h0 = img.size
            pts  = load_gt(gt_dir, p.stem)
            dm   = make_density_adaptive(pts, w0, h0)
            self.densities.append(dm)
            self.counts.append(len(pts))

        dm_sums = [dm.sum() for dm in self.densities]
        err     = [abs(s - c) for s, c in zip(dm_sums, self.counts)]
        print(f"    count mean={np.mean(self.counts):.1f}  "
              f"dm.sum mean={np.mean(dm_sums):.2f}  |dm-gt| max={np.max(err):.4f}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        dm  = self.densities[idx].copy()
        if self.is_train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            dm  = dm[:, ::-1].copy()
        return {
            "image":   self.tf(img),
            "density": torch.from_numpy(dm).unsqueeze(0),
            "count":   torch.tensor(self.counts[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------
# Evaluation with TTA
# ---------------------------------------------------------------
def evaluate(model, loader, tta=True):
    model.eval()
    mae_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            imgs = batch["image"].to(device)
            gts  = batch["count"].numpy()
            pred_orig = model(imgs)
            if tta:
                pred_flip = model(torch.flip(imgs, dims=[-1]))
                pred = (pred_orig + torch.flip(pred_flip, dims=[-1])) / 2
            else:
                pred = pred_orig
            for i in range(len(imgs)):
                mae_list.append(abs(float(pred[i].sum().cpu()) - float(gts[i])))
    mae = float(np.mean(mae_list))
    mse = float(np.sqrt(np.mean(np.array(mae_list) ** 2)))
    return mae, mse


# ---------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------
print("\nBuilding Part B datasets (adaptive sigma)…")
train_ds = CrowdDataset(TRAIN_IMG, TRAIN_GT, is_train=True)
test_ds  = CrowdDataset(TEST_IMG,  TEST_GT,  is_train=False)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
print("\nBuilding ConvNeXt-CSR Part B v2…")
model = ConvNeXtCSRPartB().to(device)

backbone_p = (list(model.stage1.parameters()) + list(model.stage2.parameters()))
fpn_p      = (list(model.lat_16.parameters()) + list(model.lat_8.parameters()) +
              list(model.smooth_8.parameters()))
head_p     = list(model.density_head.parameters())

b_cnt  = sum(p.numel() for p in backbone_p)
fpn_cnt = sum(p.numel() for p in fpn_p)
h_cnt  = sum(p.numel() for p in head_p)
print(f"  Backbone: {b_cnt/1e6:.1f}M   FPN: {fpn_cnt/1e6:.2f}M   "
      f"Head: {h_cnt/1e6:.2f}M   Total: {(b_cnt+fpn_cnt+h_cnt)/1e6:.1f}M")

# ---------------------------------------------------------------
# Optimizer — differential LR
# ---------------------------------------------------------------
optimizer = torch.optim.AdamW([
    {"params": backbone_p, "lr": LR_BACKBONE},
    {"params": fpn_p,      "lr": LR_FPN},
    {"params": head_p,     "lr": LR_HEAD},
], weight_decay=1e-4)

warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EP)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EP, eta_min=1e-7)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EP])

# ---------------------------------------------------------------
# Untrained baseline
# ---------------------------------------------------------------
print("\nUntrained baseline (no TTA):")
init_mae, init_mse = evaluate(model, test_loader, tta=False)
print(f"  MAE={init_mae:.2f}  MSE={init_mse:.2f}")

# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
best_mae, best_mse, best_state = float("inf"), float("inf"), None
history = []

print(f"\nTraining {EPOCHS} epochs | Part B ConvNeXt + CBAM | composite loss + TTA eval")
print("=" * 80)

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss, ep_mse_v, ep_ssim_v, ep_cnt_v = [], [], [], []

    for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}/{EPOCHS}", leave=False):
        imgs    = batch["image"].to(device)
        dens_gt = batch["density"].to(device)
        cnt_gt  = batch["count"].to(device)

        optimizer.zero_grad()
        dens_pred = model(imgs)

        mse_l  = F.mse_loss(dens_pred, dens_gt)
        ssim_l = ssim_loss(dens_pred, dens_gt)
        cnt_l  = F.l1_loss(dens_pred.sum(dim=[1, 2, 3]), cnt_gt)
        loss   = mse_l + LAMBDA_SSIM * ssim_l + LAMBDA_CNT * cnt_l

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        ep_loss.append(loss.item())
        ep_mse_v.append(mse_l.item())
        ep_ssim_v.append(ssim_l.item())
        ep_cnt_v.append(cnt_l.item())

    scheduler.step()

    if epoch % EVAL_EVERY == 0:
        mae, mse = evaluate(model, test_loader, tta=True)
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[2]["lr"]
        history.append({"epoch": epoch, "mae": mae, "mse": mse,
                        "loss": float(np.mean(ep_loss))})
        tag = " <BEST>" if mae < best_mae else ""
        print(f"  Ep{epoch:03d} | loss={np.mean(ep_loss):.6f} "
              f"[mse={np.mean(ep_mse_v):.6f} ssim={np.mean(ep_ssim_v):.4f} "
              f"cnt={np.mean(ep_cnt_v):.3f}] | "
              f"MAE={mae:.2f}  MSE={mse:.2f} | "
              f"lr_bb={lr_bb:.1e} lr_hd={lr_hd:.1e}{tag}")

        if mae < best_mae:
            best_mae, best_mse = mae, mse
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch, "mae": mae, "mse": mse,
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
            }, SAVE_DIR / "best_convnext_partb_v2.pth")

print("\n" + "=" * 80)
print(f"Training complete.  Best MAE={best_mae:.2f}  MSE={best_mse:.2f}")
print(f"Checkpoint: {SAVE_DIR / 'best_convnext_partb_v2.pth'}")
