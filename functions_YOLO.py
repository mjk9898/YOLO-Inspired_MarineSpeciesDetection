from sklearn.model_selection import KFold
from torch.amp import GradScaler, autocast
import torch
import csv
import os
from torch.utils.data import DataLoader
from class_YOLO import Yolo_TrainDataset, Yolo_TestDataset
import cv2
from torchvision.ops import nms


def split_dataset_kfold(image_files, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(kf.split(image_files))


def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs, fold, save_dir, patience):
    best_loss = float("inf")
    best_model_path = os.path.join(save_dir, f"best_model_fold{fold}.pth")

    history = []  # List of dicts with keys: epoch, train_loss, val_loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        scaler = GradScaler(device='cuda')

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                preds = model(imgs)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        if (epoch + 1) % 10 == 0:
            print(
                f"[Fold {fold}] Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                print(
                    f"[Fold {fold}] Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                print(f"Best Validation Loss: {best_loss}")
                break

    # Save history to CSV
    csv_path = os.path.join(save_dir, f"loss_history_fold{fold}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    return best_model_path


def run_kfold_training(image_dir, label_dir, model_class, num_classes, criterion, k=5, batch_size=16, epochs=50,
                       save_dir="weights", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    # Load image file names once
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    data_folds = split_dataset_kfold(image_files, k=k)

    for fold in range(k):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        train_idx, val_idx = data_folds[fold]

        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]

        # Augmented training dataset
        train_dataset = Yolo_TrainDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            sampling_prob=None,
            grid_size=(19, 50),
            num_classes=num_classes
        )
        train_dataset.image_files = train_files  # overwrite with fold-specific images

        # Non-augmented val dataset
        val_dataset = Yolo_TestDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            grid_size=(19, 50),
            num_classes=num_classes
        )
        val_dataset.image_files = val_files

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Change Num_workers based on GPU
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model & optimizer
        model = model_class().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=5e-4)

        # Train one fold
        train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            fold=fold + 1,
            save_dir=save_dir,
            patience=20
        )

##################################################################################

def decode_predictions_with_nms(preds, conf_thresh=0.5, iou_thresh=0.5, img_size=(300, 800)):
    preds = preds.detach().cpu().numpy()
    boxes, scores, class_ids = [], [], []
    grid_h, grid_w = preds.shape[:2]
    img_h, img_w = img_size

    for i in range(grid_h):
        for j in range(grid_w):
            pred = preds[i, j]
            x, w = pred[:2]  # x_center and width only
            conf = torch.sigmoid(torch.tensor(pred[2]))
            class_logits = torch.tensor(pred[3:])
            class_probs = torch.sigmoid(class_logits)
            class_id = torch.argmax(class_probs).item()
            class_conf = class_probs[class_id].item()
            total_conf = conf.item() * class_conf

            if total_conf >= conf_thresh:
                abs_x = (j + x) / grid_w * img_w
                abs_w = w * img_w
                abs_y = img_h * 0.5
                abs_h = img_h * 1.0
                x1 = abs_x - abs_w / 2
                y1 = abs_y - abs_h / 2
                x2 = abs_x + abs_w / 2
                y2 = abs_y + abs_h / 2
                boxes.append([x1, y1, x2, y2])
                scores.append(total_conf)
                class_ids.append(class_id)

    if not boxes:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    class_ids = torch.tensor(class_ids)

    final_boxes = []
    for class_label in class_ids.unique():
        inds = (class_ids == class_label).nonzero(as_tuple=True)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        keep = nms(cls_boxes, cls_scores, iou_thresh)

        for i in keep:
            x1, y1, x2, y2 = cls_boxes[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            final_boxes.append([
                cx.item(), cy.item(), w.item(), h.item(),
                cls_scores[i].item(), int(class_label)
            ])

    return final_boxes

def load_models(model_class, weights_dir, device, num_folds=5):
    models = []
    for i in range(1, num_folds + 1):
        model = model_class().to(device)
        weights_path = os.path.join(weights_dir, f"best_model_fold{i}.pth")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    return models

def ensemble_predict(models, image_tensor):
    outputs = [model(image_tensor)[0].detach().cpu() for model in models]
    avg_output = torch.stack(outputs).mean(dim=0)  # (H, W, C)
    return avg_output


def predict_ensemble(image_path, models, device):
    # Load image and preprocess
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (800, 300))
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Ensemble prediction
    avg_pred = ensemble_predict(models, img_tensor)
    boxes = decode_predictions_with_nms(avg_pred, conf_thresh=0.5, iou_thresh=0.5, img_size=(300, 800))
    return boxes

def save_yolo_preds(pred_boxes, save_path, img_size=(300, 800)):
    h, w = img_size
    pred_boxes = sorted(pred_boxes, key=lambda box: int(box[5]))
    with open(save_path, "w") as f:
        for box in pred_boxes:
            cx, cy, bw, bh, conf, class_id = box
            x = cx / w
            y = cy / h
            w_norm = bw / w
            h_norm = bh / h
            f.write(f"{class_id} {x:.6f} {y:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.3f}\n")

##############################################################################

def parse_yolo_pred_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else None
    return {
        "class_id": cls,
        "x_center": x_center,
        "width": width,
        "conf": conf
    }

def yolo_to_times(rows, total_duration):
    out = []
    for r in rows:
        x, w = r["x_center"], r["width"]
        t0 = max(0.0, (x - w/2.0) * total_duration)
        t1 = min(total_duration, (x + w/2.0) * total_duration)
        out.append({
            "class_id": r["class_id"],
            "start_time_sec": t0,
            "end_time_sec": t1,
            "duration_sec": max(0.0, t1 - t0),
            "conf": r["conf"]
        })
    # optional: sort by start time
    out.sort(key=lambda d: d["start_time_sec"])
    return out

def convert_yolo_file(txt_path, total_duration, file_id=None):
    with open(txt_path, "r", encoding="utf-8") as f:
        rows = [parse_yolo_pred_line(line) for line in f if line.strip()]
    rows = [r for r in rows if r is not None]
    recs = yolo_to_times(rows, total_duration)
    for r in recs:
        r["source_file"] = file_id if file_id is not None else os.path.basename(txt_path)
    return recs
