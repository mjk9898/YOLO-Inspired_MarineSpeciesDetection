import os
from functions_YOLO import load_models, predict_ensemble, save_yolo_preds, convert_yolo_file
from class_YOLO import YOLO_EfficientNet
import torch
import glob
import pandas as pd

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Predction using YOLO
weights_dir = "YOLO-cv_weights"
test_images_dir = "YOLO-TestData/spectrograms"
output_label_dir = "YOLO-TestData/predictions_YOLO"
os.makedirs(output_label_dir, exist_ok=True)

# Load all models
# Types of Species to be Detected
with open("YOLO-TrainData/species.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

num_classes=len(class_names)
models = load_models(lambda: YOLO_EfficientNet(num_classes=num_classes), weights_dir, device, num_folds=5)

# Loop over test images
for file in sorted(os.listdir(test_images_dir)):
    if not file.endswith(".png"):
        continue

    img_path = os.path.join(test_images_dir, file)
    pred_boxes = predict_ensemble(img_path, models, device)

    label_name = os.path.splitext(file)[0] + ".txt"
    save_path = os.path.join(output_label_dir, label_name)
    save_yolo_preds(pred_boxes, save_path)

#######################################################################

# Save Prediction Durations for each Spectrogram

class_map= {}
with open("YOLO-TrainData/species_label_mapping.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        idx_str, name = line.split(":", 1)
        idx = int(idx_str.strip())
        name = name.strip()

        class_map[idx] = name

total_duration = float(input("Enter Time Window used for Spectrogram (sec): "))

all_recs = []
for txt_path in glob.glob("YOLO-TestData/predictions_YOLO/*.txt"):
    recs = convert_yolo_file(txt_path, total_duration=total_duration)
    for r in recs:
        r["class_name"] = class_map.get(r["class_id"], "Unknown")
    all_recs.extend(recs)

df = pd.DataFrame(all_recs)
df.to_csv("YOLO-TestData/predictions_test_with_times.csv", index=False)

print(f"Converted {len(all_recs)} predictions from {len(glob.glob('YOLO-TestData/predictions_YOLO/*.txt'))} files.")