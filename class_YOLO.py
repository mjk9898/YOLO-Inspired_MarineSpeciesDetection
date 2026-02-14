import torch.nn as nn
import torch
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import random
import numpy as np
import os
import cv2
from torch.utils.data import Dataset

# --- Depthwise Separable Convolution ---
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


# --- Dilated Temporal Convolution Block ---
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
            nn.GroupNorm(4, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


# --- C2f Block (YOLOv8-style CSP variant) ---
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = nn.Conv2d(in_channels, hidden, 1, 1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(4, hidden),
                nn.SiLU()
            ) for _ in range(num_blocks)
        ])
        self.cv2 = nn.Conv2d(hidden * (num_blocks + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        chunks = [x]
        for block in self.blocks:
            x = block(x)
            chunks.append(x)
        return self.cv2(torch.cat(chunks, dim=1))


# --- SPPF Module ---
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        x = torch.cat([x, y1, y2, y3], dim=1)
        return self.act(self.conv2(x))


# --- Decoupled Detection Head (modified for time-only box prediction) ---
class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)

        self.obj_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.obj_out = nn.Conv2d(in_channels, 1, 1)

        self.box_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.box_out = nn.Conv2d(in_channels, 2, 1)  # Only x_center and width

    def forward(self, x):
        cls = self.cls_out(F.silu(self.cls_conv(x)))
        obj = self.obj_out(F.silu(self.obj_conv(x)))
        box = self.box_out(F.silu(self.box_conv(x)))
        return torch.cat([box, obj, cls], dim=1)


# --- Full YOLOv8-style Model with Multi-Scale Fusion & Temporal Convs ---
class YOLO_EfficientNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.output_channels = 3 + num_classes

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.backbone = create_feature_extractor(
            backbone,
            return_nodes={
                'features.3': 'feat2',
                'features.4': 'feat3',
                'features.6': 'feat4'
            }
        )

        # Channel reductions
        self.reduce2 = nn.Conv2d(48, 48, 1)
        self.reduce3 = nn.Conv2d(96, 48, 1)
        self.reduce4 = nn.Conv2d(232, 48, 1)

        # Top-down FPN
        self.c2f3 = C2f(96, 48)
        self.c2f2 = C2f(96, 48)

        # Bottom-up PAN
        self.downsample2 = DWConv(48, 48)
        self.pan_c2f3 = C2f(96, 48)

        # Temporal Convs
        self.temp1 = TemporalConvBlock(48, 48, dilation=1)
        self.temp2 = TemporalConvBlock(48, 48, dilation=2)

        # SPPF
        self.sppf = SPPF(48, 48)

        # Multi-scale Heads
        self.head1 = DecoupledHead(48, num_classes)
        self.head2 = DecoupledHead(48, num_classes)

        self.fuse = nn.Conv2d(48 * 2, 48, 1)

    def forward(self, x):
        feats = self.backbone(x)
        f2 = self.reduce2(feats['feat2'])
        f3 = self.reduce3(feats['feat3'])
        f4 = self.reduce4(feats['feat4'])

        # Top-down
        up3 = F.interpolate(f4, size=f3.shape[2:], mode='nearest')
        x3 = self.c2f3(torch.cat([up3, f3], dim=1))

        up2 = F.interpolate(x3, size=f2.shape[2:], mode='nearest')
        x2 = self.c2f2(torch.cat([up2, f2], dim=1))

        # Bottom-up
        down2 = self.downsample2(x2)
        if down2.shape[2:] != x3.shape[2:]:
            down2 = F.interpolate(down2, size=x3.shape[2:], mode='nearest')

        pan = self.pan_c2f3(torch.cat([down2, x3], dim=1))

        # Temporal convs
        t1 = self.temp1(pan)
        t2 = self.temp2(pan)

        fused = self.fuse(torch.cat([t1, t2], dim=1))

        # Head outputs
        head1_out = self.head1(self.sppf(pan))
        head2_out = self.head2(self.sppf(fused))

        out = (head1_out + head2_out) / 2.0
        out = out.permute(0, 2, 3, 1).contiguous()
        return out

#############################################################################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_term = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'sum':
            return focal_term.sum()
        elif self.reduction == 'mean':
            return focal_term.mean()
        else:
            return focal_term


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, lambda_box=0.05, lambda_obj=1.0, lambda_cls=1.0,
                 use_focal=True, alpha=0.25, gamma=1.5, label_smoothing=0.01):   ## change hyperparameters as needed
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='sum')

    def forward(self, preds, targets):
        object_mask = targets[..., 2] == 1
        noobj_mask = targets[..., 2] == 0

        # === Box Loss (x_center and width only) ===
        pred_box = preds[..., 0:2][object_mask]
        true_box = targets[..., 0:2][object_mask]

        if pred_box.shape[0] == 0:
            box_loss = torch.tensor(0.0, device=preds.device)
        else:
            box_loss = F.smooth_l1_loss(pred_box, true_box, reduction='sum')

        # === Objectness Loss ===
        obj_pred = preds[..., 2]
        obj_true = targets[..., 2]

        if self.use_focal:
            objectness_loss = self.focal(obj_pred[object_mask], obj_true[object_mask]) + \
                              0.5 * self.focal(obj_pred[noobj_mask], obj_true[noobj_mask])
        else:
            obj_loss = self.bce(obj_pred[object_mask], obj_true[object_mask])
            noobj_loss = self.bce(obj_pred[noobj_mask], obj_true[noobj_mask])
            objectness_loss = obj_loss + 0.5 * noobj_loss

        # === Classification Loss ===
        cls_pred = preds[..., 3:][object_mask]
        cls_true = targets[..., 3:][object_mask]

        if self.label_smoothing > 0:
            cls_true = cls_true * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes

        if self.use_focal:
            cls_loss = self.focal(cls_pred, cls_true)
        else:
            cls_loss = self.bce(cls_pred, cls_true)

        total_loss = (
            self.lambda_box * box_loss +
            self.lambda_obj * objectness_loss +
            self.lambda_cls * cls_loss
        )

        return total_loss

##########################################################################################

class Yolo_TestDataset(Dataset):
    def __init__(self, image_dir, label_dir, grid_size=(19, 50), num_classes=17, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.grid_h, self.grid_w = grid_size
        self.num_classes = num_classes
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 300))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        target = torch.zeros((self.grid_h, self.grid_w, 3 + self.num_classes))

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gx = int(x * self.grid_w)
                    gy = int(y * self.grid_h)
                    if gx >= self.grid_w or gy >= self.grid_h:
                        continue
                    target[gy, gx, 0:2] = torch.tensor([x, w])
                    target[gy, gx, 2] = 1.0
                    target[gy, gx, 3 + int(class_id)] = 1.0

        return image, target


###################################

class Yolo_TrainDataset(Yolo_TestDataset):
    def __init__(self, *args, sampling_prob=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_prob = sampling_prob or {}

    def cutmix(self, base_img, base_target, patch_img, patch_target, x_offset):

        H, W = base_img.shape[:2]
        new_img = base_img.copy()

        has_yh = (patch_target.shape[-1] >= 5)
        obj_idx = 4 if has_yh else 2
        cls_from = 5 if has_yh else 3

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if float(patch_target[gy, gx, obj_idx]) != 1.0:
                    continue

                tail = patch_target[gy, gx, obj_idx:].clone()  # obj + classes (we'll reuse)

                if has_yh:
                    x_c, y_c, w_n, h_n = patch_target[gy, gx, 0:4].tolist()

                    # bbox in patch pixels
                    x_c_px = x_c * W;
                    y_c_px = y_c * H
                    half_w = (w_n * W) / 2.0
                    half_h = (h_n * H) / 2.0
                    src_x0 = int(np.floor(x_c_px - half_w))
                    src_x1 = int(np.ceil(x_c_px + half_w))
                    src_y0 = int(np.floor(y_c_px - half_h))
                    src_y1 = int(np.ceil(y_c_px + half_h))

                    # clip to patch
                    src_x0 = max(0, src_x0);
                    src_x1 = min(W, src_x1)
                    src_y0 = max(0, src_y0);
                    src_y1 = min(H, src_y1)
                    if src_x1 <= src_x0 or src_y1 <= src_y0:
                        continue

                    # destination after shift
                    dst_x0 = src_x0 + x_offset
                    dst_x1 = src_x1 + x_offset
                    dst_y0 = src_y0
                    dst_y1 = src_y1

                    # horizontal clipping at destination (adjust source to match)
                    if dst_x0 < 0:
                        shift = -dst_x0
                        src_x0 += shift;
                        dst_x0 = 0
                    if dst_x1 > W:
                        shrink = dst_x1 - W
                        src_x1 -= shrink;
                        dst_x1 = W
                    if dst_x1 <= dst_x0:
                        continue

                    # paste bbox rectangle only (non-destructive elsewhere)
                    new_img[dst_y0:dst_y1, dst_x0:dst_x1] = patch_img[src_y0:src_y1, src_x0:src_x1]

                    # new normalized center & size (accounting for horizontal clipping)
                    vis_w_px = float(dst_x1 - dst_x0)
                    vis_h_px = float(dst_y1 - dst_y0)
                    x_new = float(np.clip((x_c_px + x_offset) / W, 0.0, 1.0))
                    y_new = float(np.clip(y_c_px / H, 0.0, 1.0))
                    w_new = max(1.0, vis_w_px) / float(W)
                    h_new = max(1.0, vis_h_px) / float(H)

                    gx_new = int(x_new * self.grid_w)
                    gy_new = int(y_new * self.grid_h)
                    if not (0 <= gx_new < self.grid_w and 0 <= gy_new < self.grid_h):
                        continue

                    # --- Non-destructive write: only if destination cell is empty
                    if float(base_target[gy_new, gx_new, obj_idx]) != 1.0:
                        base_target[gy_new, gx_new, 0] = x_new
                        base_target[gy_new, gx_new, 1] = y_new
                        base_target[gy_new, gx_new, 2] = w_new
                        base_target[gy_new, gx_new, 3] = h_new
                        base_target[gy_new, gx_new, 4:] = tail
                    # else: occupied → skip (or resolve below)

                else:
                    # legacy: [x, w, obj, one-hot...] assume full-height, bottom-anchored bbox
                    x_c, w_n = patch_target[gy, gx, 0:2].tolist()
                    x_c_px = x_c * W
                    half_w = (w_n * W) / 2.0
                    src_x0 = int(np.floor(x_c_px - half_w))
                    src_x1 = int(np.ceil(x_c_px + half_w))
                    src_y0, src_y1 = 0, H

                    src_x0 = max(0, src_x0);
                    src_x1 = min(W, src_x1)
                    if src_x1 <= src_x0:
                        continue

                    dst_x0 = src_x0 + x_offset
                    dst_x1 = src_x1 + x_offset
                    if dst_x0 < 0:
                        shift = -dst_x0
                        src_x0 += shift;
                        dst_x0 = 0
                    if dst_x1 > W:
                        shrink = dst_x1 - W
                        src_x1 -= shrink;
                        dst_x1 = W
                    if dst_x1 <= dst_x0:
                        continue

                    new_img[src_y0:src_y1, dst_x0:dst_x1] = patch_img[src_y0:src_y1, src_x0:src_x1]

                    vis_w_px = float(dst_x1 - dst_x0)
                    x_new = float(np.clip((x_c_px + x_offset) / W, 0.0, 1.0))
                    w_new = max(1.0, vis_w_px) / float(W)

                    gx_new = int(x_new * self.grid_w)
                    if not (0 <= gx_new < self.grid_w):
                        continue

                    # Non-destructive label add if empty
                    if float(base_target[gy, gx_new, 2]) != 1.0:
                        base_target[gy, gx_new, 0] = x_new
                        base_target[gy, gx_new, 1] = w_new
                        base_target[gy, gx_new, 2:] = tail
                    # else: occupied → skip (or resolve)

        return new_img, base_target

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_np = image.permute(1, 2, 0).numpy().copy()

        # === Class-based CutMix ===
        if self.sampling_prob:
            class_map = torch.argmax(target[..., 3:], dim=-1)
            presence_mask = target[..., 2] == 1
            present_classes = class_map[presence_mask]
            should_cutmix = any(random.random() < self.sampling_prob.get(int(cls), 0.5) for cls in present_classes)

            if should_cutmix:
                idx2 = random.randint(0, len(self.image_files) - 1)
                image2, target2 = super().__getitem__(idx2)
                image2_np = image2.permute(1, 2, 0).numpy().copy()
                x_offset = random.randint(100, 700)
                image_np, target = self.cutmix(image_np, target, image2_np, target2, x_offset)

        # === Random Horizontal Flip ===
        if random.random() < 0.5:
            image_np = cv2.flip(image_np, 1)
            for gy in range(self.grid_h):
                for gx in range(self.grid_w):
                    if target[gy, gx, 2] == 1.0:
                        x, w = target[gy, gx, 0:2]
                        target[gy, gx, 0] = 1.0 - x

        # === Brightness / Contrast ===
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-0.1, 0.1)
            image_np = np.clip(alpha * image_np + beta, 0, 1)

        # === Hue/Saturation Jitter ===
        if random.random() < 0.5:
            hsv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hue_shift = random.randint(-10, 10)
            sat_mult = random.uniform(0.9, 1.1)
            val_mult = random.uniform(0.9, 1.1)
            hsv = hsv.astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
            hsv[..., 1] *= sat_mult
            hsv[..., 2] *= val_mult
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0

        image = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
        return image, target
