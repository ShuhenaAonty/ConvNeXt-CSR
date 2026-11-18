"""
ConvNeXt-CSR  EMA + SGDR  –  ShanghaiTech Part A
=================================================
Baseline (convnext_crop_parta.py): MAE=62.70 @ Ep160

Two targeted fixes for the post-Ep160 degradation:

  1. EMA (decay=0.999 per step)
       Eval uses shadow-average weights instead of current weights.
       Smooths out weight oscillations; empirically adds 1-3 MAE on top
       of any training-time gain without extra compute.

  2. SGDR – CosineAnnealingWarmRestarts(T_0=100)
       After 20-ep warmup the LR cycles 100→0 repeatedly.
       Restarts at Ep120, 220, 320, 420 → model re-explores the
       low-loss basin every cycle instead of dying at lr≈1e-7.

Everything else is identical to the baseline (architecture, loss,
crop strategy, augmentation, eval TTA scales).
"""

import copy, sys, math, warnings, random
warnings.filterwarnings("ignore")
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

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"D:\Research\Crowd Counting\Dataset\ShanghaiTech\part_A")
TRAIN_IMG = DATASET_ROOT / "train_data" / "images"
TRAIN_GT  = DATASET_ROOT / "train_data" / "ground-truth"
TEST_IMG  = DATASET_ROOT / "test_data"  / "images"
TEST_GT   = DATASET_ROOT / "test_data"  / "ground-truth"

BASE_DIR = Path(r"D:\Research\Crowd Counting\Code\crowd_counting_notebooks_and_modules")
LOG_FILE = BASE_DIR / "convnext_ema_sgdr_parta_log.txt"
SAVE_DIR = BASE_DIR / "final files" / "checkpoints_convnext_ema_sgdr_parta"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

import io
class _Tee(io.TextIOWrapper):
    def __init__(self, path, original):
        self._f = open(path, "w", encoding="utf-8")
        self._orig = original
    def write(self, s):
        self._f.write(s); self._f.flush()
        self._orig.write(s); self._orig.flush()
        return len(s)
    def flush(self):
        self._f.flush(); self._orig.flush()
sys.stdout = _Tee(LOG_FILE, sys.__stdout__)

# ─────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────
CROP_SIZE   = 512
OUT_CROP    = CROP_SIZE // 8      # 64×64 density map
SCALE_MIN   = 0.7
SCALE_MAX   = 1.3
K_NEIGHBORS = 4
BETA        = 0.3
SIGMA_MIN   = 1.0
SIGMA_MAX   = 15.0
LAMBDA_SSIM = 0.1
LAMBDA_CNT  = 0.01
EPOCHS      = 500
WARMUP_EP   = 20
BATCH_SIZE  = 8
LR_BACKBONE = 5e-6
LR_FPN      = 2e-5
LR_HEAD     = 5e-5
ETA_MIN     = 1e-7                # cosine floor
SGDR_T0     = 100                 # restart period (epochs after warmup)
EMA_DECAY   = 0.999               # per optimizer-step
EVAL_SCALES = [0.75, 1.0, 1.25]
EVAL_EVERY  = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"ConvNeXt EMA+SGDR Part A | crop={CROP_SIZE} scale=[{SCALE_MIN},{SCALE_MAX}]")
print(f"LR backbone={LR_BACKBONE} fpn={LR_FPN} head={LR_HEAD} | ETA_MIN={ETA_MIN}")
print(f"Epochs={EPOCHS} warmup={WARMUP_EP} SGDR_T0={SGDR_T0} EMA_DECAY={EMA_DECAY}")
print(f"Eval TTA scales: {EVAL_SCALES}")


# ─────────────────────────────────────────────────────────────────
# LR schedule: linear warmup then SGDR (no scheduler object needed)
# ─────────────────────────────────────────────────────────────────
def get_lr(epoch: int, base_lr: float) -> float:
    """0-indexed epoch → LR for that epoch."""
    if epoch < WARMUP_EP:
        return base_lr * (0.05 + 0.95 * (epoch + 1) / WARMUP_EP)
    t = (epoch - WARMUP_EP) % SGDR_T0
    return ETA_MIN + 0.5 * (base_lr - ETA_MIN) * (1.0 + math.cos(math.pi * t / SGDR_T0))

def set_lr(optimizer, epoch):
    optimizer.param_groups[0]["lr"] = get_lr(epoch, LR_BACKBONE)
    optimizer.param_groups[1]["lr"] = get_lr(epoch, LR_FPN)
    optimizer.param_groups[2]["lr"] = get_lr(epoch, LR_HEAD)


# ─────────────────────────────────────────────────────────────────
# EMA helpers
# ─────────────────────────────────────────────────────────────────
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ─────────────────────────────────────────────────────────────────
# SSIM loss
# ─────────────────────────────────────────────────────────────────
def _gaussian_kernel_2d(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    return g2d.unsqueeze(0).unsqueeze(0)

def ssim_loss(pred, target, kernel_size=11, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma, device=pred.device)
    pad    = kernel_size // 2
    mu_x   = F.conv2d(pred,   kernel, padding=pad)
    mu_y   = F.conv2d(target, kernel, padding=pad)
    mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
    mu_xy  = mu_x * mu_y
    sig_x2 = F.conv2d(pred   ** 2,   kernel, padding=pad) - mu_x2
    sig_y2 = F.conv2d(target ** 2,   kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(pred * target, kernel, padding=pad) - mu_xy
    numer  = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denom  = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return 1.0 - (numer / (denom + 1e-8)).mean()


# ─────────────────────────────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


# ─────────────────────────────────────────────────────────────────
# Model: ConvNeXt-Tiny + 2-scale FPN + CBAM  (identical to baseline)
# ─────────────────────────────────────────────────────────────────
class ConvNeXtCropPartA(nn.Module):
    def __init__(self):
        super().__init__()
        cnxt = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        features = list(cnxt.features.children())
        self.stage1 = nn.Sequential(*features[:4])   # H/8,  C=192
        self.stage2 = nn.Sequential(*features[4:6])  # H/16, C=384
        self.lat_16 = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.lat_8  = nn.Sequential(
            nn.Conv2d(192, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.smooth_8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            CBAM(256),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            CBAM(128),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        for mod in [self.lat_16, self.lat_8, self.smooth_8, self.head]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None: nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        f8  = self.stage1(x)
        f16 = self.stage2(f8)
        p16 = self.lat_16(f16)
        p8  = self.lat_8(f8) + F.interpolate(p16, size=f8.shape[2:], mode="bilinear", align_corners=False)
        p8  = self.smooth_8(p8)
        return F.relu(self.head(p8))


# ─────────────────────────────────────────────────────────────────
# GT loading + density map generation  (identical to baseline)
# ─────────────────────────────────────────────────────────────────
def load_gt(gt_dir, stem):
    mat = sio.loadmat(Path(gt_dir) / f"GT_{stem}.mat")
    try:
        pts = mat["image_info"][0, 0][0, 0][0]
    except Exception:
        pts = mat.get("annPoints", np.zeros((0, 2)))
    return pts.astype(np.float32)

def make_density_from_pts(pts_dm, out_h, out_w):
    dm = np.zeros((out_h, out_w), dtype=np.float32)
    n = len(pts_dm)
    if n == 0: return dm
    if n == 1:
        xi = int(np.clip(pts_dm[0, 0], 0, out_w - 1))
        yi = int(np.clip(pts_dm[0, 1], 0, out_h - 1))
        single = np.zeros((out_h, out_w), dtype=np.float64)
        single[yi, xi] = 1.0
        bl = gaussian_filter(single, sigma=4.0)
        s  = bl.sum()
        if s > 0: dm += (bl / s).astype(np.float32)
        return dm
    tree     = cKDTree(pts_dm)
    k_actual = min(K_NEIGHBORS + 1, n)
    dists, _ = tree.query(pts_dm, k=k_actual)
    sigmas   = np.clip(BETA * dists[:, 1:].mean(axis=1), SIGMA_MIN, SIGMA_MAX)
    for i, pt in enumerate(pts_dm):
        xi = int(np.clip(pt[0], 0, out_w - 1))
        yi = int(np.clip(pt[1], 0, out_h - 1))
        single = np.zeros((out_h, out_w), dtype=np.float64)
        single[yi, xi] = 1.0
        bl = gaussian_filter(single, sigma=float(sigmas[i]))
        s  = bl.sum()
        if s > 0: bl /= s
        dm += bl.astype(np.float32)
    return dm


# ─────────────────────────────────────────────────────────────────
# ImageNet normalisation + augmentation
# ─────────────────────────────────────────────────────────────────
NORM      = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_TENSOR = T.ToTensor()
COLOR_JIT = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GRAY_AUG  = T.RandomGrayscale(p=0.05)


# ─────────────────────────────────────────────────────────────────
# Crop Training Dataset  (identical to baseline)
# ─────────────────────────────────────────────────────────────────
class CropTrainDataset(Dataset):
    CROPS_PER_IMAGE = 8

    def __init__(self, img_dir, gt_dir):
        self.img_paths = sorted(Path(img_dir).glob("*.jpg"))
        self.gt_pts = {p.stem: load_gt(gt_dir, p.stem) for p in self.img_paths}
        print(f"  [Train] {len(self.img_paths)} images x {self.CROPS_PER_IMAGE} crops "
              f"= {len(self)} samples/epoch")

    def __len__(self):
        return len(self.img_paths) * self.CROPS_PER_IMAGE

    def __getitem__(self, idx):
        img_path = self.img_paths[idx % len(self.img_paths)]
        pts_orig = self.gt_pts[img_path.stem].copy()
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        scale  = random.uniform(SCALE_MIN, SCALE_MAX)
        new_w  = max(CROP_SIZE, int(round(w0 * scale)))
        new_h  = max(CROP_SIZE, int(round(h0 * scale)))
        img_sc = img.resize((new_w, new_h), Image.BILINEAR)
        pts_sc = pts_orig.copy()
        if len(pts_sc) > 0:
            pts_sc[:, 0] *= new_w / w0
            pts_sc[:, 1] *= new_h / h0

        ox = random.randint(0, new_w - CROP_SIZE)
        oy = random.randint(0, new_h - CROP_SIZE)
        img_crop = img_sc.crop((ox, oy, ox + CROP_SIZE, oy + CROP_SIZE))

        if len(pts_sc) > 0:
            px   = pts_sc[:, 0] - ox
            py   = pts_sc[:, 1] - oy
            mask = (px >= 0) & (px < CROP_SIZE) & (py >= 0) & (py < CROP_SIZE)
            pts_crop = np.column_stack([px[mask], py[mask]]) if mask.sum() > 0 else np.zeros((0, 2))
        else:
            pts_crop = np.zeros((0, 2))

        dm_scl = OUT_CROP / CROP_SIZE
        pts_dm = pts_crop * dm_scl if len(pts_crop) > 0 else pts_crop
        dm     = make_density_from_pts(pts_dm, OUT_CROP, OUT_CROP)
        count  = float(len(pts_crop))

        if random.random() < 0.5:
            img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if len(pts_crop) > 0:
                dm = dm[:, ::-1].copy()

        img_crop = COLOR_JIT(img_crop)
        img_crop = GRAY_AUG(img_crop)

        return {
            "image":   NORM(TO_TENSOR(img_crop)),
            "density": torch.from_numpy(dm.copy()).unsqueeze(0),
            "count":   torch.tensor(count, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────
# Multi-scale Evaluation with H-flip TTA  (runs on whichever model is passed)
# ─────────────────────────────────────────────────────────────────
def evaluate(model, test_img_dir, test_gt_dir):
    model.eval()
    img_paths = sorted(Path(test_img_dir).glob("*.jpg"))
    errors = []
    with torch.no_grad():
        for p in tqdm(img_paths, desc="  eval", leave=False):
            pts      = load_gt(test_gt_dir, p.stem)
            gt_count = float(len(pts))
            img      = Image.open(p).convert("RGB")
            w0, h0   = img.size
            preds    = []
            for s in EVAL_SCALES:
                nw = int(round(w0 * s / 8) * 8); nh = int(round(h0 * s / 8) * 8)
                nw, nh = max(nw, 8), max(nh, 8)
                img_r = img.resize((nw, nh), Image.BILINEAR)
                t     = NORM(TO_TENSOR(img_r)).unsqueeze(0).to(device)
                preds.append(float(model(t).sum().cpu()))
                preds.append(float(model(torch.flip(t, [3])).sum().cpu()))
            errors.append(abs(np.mean(preds) - gt_count))
    mae = float(np.mean(errors))
    mse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    return mae, mse


# ─────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────
print("\nBuilding crop training dataset...")
train_ds     = CropTrainDataset(TRAIN_IMG, TRAIN_GT)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

print("\nBuilding model...")
model = ConvNeXtCropPartA().to(device)
total_p = sum(p.numel() for p in model.parameters())
bb_p    = sum(p.numel() for p in list(model.stage1.parameters()) + list(model.stage2.parameters()))
print(f"  Total: {total_p/1e6:.1f}M  Backbone: {bb_p/1e6:.1f}M  New: {(total_p-bb_p)/1e6:.1f}M")

# EMA shadow model — starts as an exact copy, stays on same device
ema_model = copy.deepcopy(model)
for p in ema_model.parameters():
    p.requires_grad_(False)
ema_model.eval()
print(f"  EMA model created (decay={EMA_DECAY})")

optimizer = torch.optim.AdamW([
    {"params": list(model.stage1.parameters()) + list(model.stage2.parameters()), "lr": LR_BACKBONE},
    {"params": list(model.lat_16.parameters()) + list(model.lat_8.parameters()) +
               list(model.smooth_8.parameters()), "lr": LR_FPN},
    {"params": list(model.head.parameters()), "lr": LR_HEAD},
], weight_decay=1e-4)

# No PyTorch scheduler — LR is set manually each epoch via get_lr()
print(f"  SGDR: warmup {WARMUP_EP} ep → cosine restarts every {SGDR_T0} ep")
print(f"  Restart epochs (post-warmup): {list(range(WARMUP_EP + SGDR_T0, EPOCHS + 1, SGDR_T0))}")


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────
print("\nUntrained baseline (EMA model, multi-scale TTA):")
init_mae, init_mse = evaluate(ema_model, TEST_IMG, TEST_GT)
print(f"  MAE={init_mae:.2f}  MSE={init_mse:.2f}")

best_mae, best_mse = float("inf"), float("inf")
print(f"\nTraining {EPOCHS} ep | EMA+SGDR | crop={CROP_SIZE} | FPN+CBAM | MS-TTA eval")
print("=" * 80)

for epoch in range(1, EPOCHS + 1):
    # Set LR for this epoch (warmup or SGDR)
    set_lr(optimizer, epoch - 1)   # epoch-1 → 0-indexed

    model.train()
    ep_loss, ep_mse_l, ep_ssim_l, ep_cnt_l = [], [], [], []

    for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}/{EPOCHS}", leave=False):
        imgs     = batch["image"].to(device)
        dens_gt  = batch["density"].to(device)
        cnt_gt   = batch["count"].to(device)

        optimizer.zero_grad()
        dens_pred = model(imgs)
        mse_l     = F.mse_loss(dens_pred, dens_gt)
        ssim_l    = ssim_loss(dens_pred, dens_gt)
        cnt_pred  = dens_pred.sum(dim=[1, 2, 3])
        cnt_l     = F.l1_loss(cnt_pred, cnt_gt)
        loss      = mse_l + LAMBDA_SSIM * ssim_l + LAMBDA_CNT * cnt_l
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # EMA update after every optimizer step
        update_ema(ema_model, model, EMA_DECAY)

        ep_loss.append(loss.item()); ep_mse_l.append(mse_l.item())
        ep_ssim_l.append(ssim_l.item()); ep_cnt_l.append(cnt_l.item())

    if epoch % EVAL_EVERY == 0:
        # Evaluate EMA model (not the live model)
        mae, mse = evaluate(ema_model, TEST_IMG, TEST_GT)
        lr_bb = optimizer.param_groups[0]["lr"]
        tag   = " <BEST>" if mae < best_mae else ""
        # Note whether this epoch is a SGDR restart
        ep_post_warmup = epoch - WARMUP_EP
        restart_tag = " [RESTART]" if ep_post_warmup > 0 and ep_post_warmup % SGDR_T0 == 0 else ""
        print(f"  Ep{epoch:03d} | loss={np.mean(ep_loss):.6f} "
              f"[mse={np.mean(ep_mse_l):.5f} ssim={np.mean(ep_ssim_l):.4f} "
              f"cnt={np.mean(ep_cnt_l):.2f}] | "
              f"MAE={mae:.2f} MSE={mse:.2f} | lr_bb={lr_bb:.1e}{tag}{restart_tag}")
        if mae < best_mae:
            best_mae, best_mse = mae, mse
            torch.save({
                "epoch": epoch, "mae": mae, "mse": mse,
                "ema_state":   ema_model.state_dict(),
                "model_state": model.state_dict(),
                "config": {"crop_size": CROP_SIZE, "eval_scales": EVAL_SCALES,
                           "ema_decay": EMA_DECAY, "sgdr_t0": SGDR_T0},
            }, SAVE_DIR / "best_convnext_ema_sgdr_parta.pth")

print(f"\nTraining complete.")
print(f"  Best EMA MAE={best_mae:.2f}  MSE={best_mse:.2f}")
print(f"  Checkpoint: {SAVE_DIR / 'best_convnext_ema_sgdr_parta.pth'}")
