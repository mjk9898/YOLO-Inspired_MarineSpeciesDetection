from functions_YOLO import run_kfold_training
from class_YOLO import YOLOLoss, YOLO_EfficientNet
import torch

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Types of Species to be Detected
with open("YOLO-TrainData/species.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Train Model
num_classes=len(class_names)
criterion = YOLOLoss(num_classes=num_classes)

run_kfold_training(
    image_dir="YOLO-TrainData/spectrogram",
    label_dir="YOLO-TrainData/annotations",
    model_class=lambda: YOLO_EfficientNet(num_classes=num_classes),
    num_classes=num_classes,
    criterion=criterion,
    k=5,
    batch_size=16,
    epochs=300,
    save_dir="YOLO-cv_weights",
    device=device
)
