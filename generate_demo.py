import cv2
import os
import glob
import random
from pathlib import Path

# Create assets directory
os.makedirs("assets", exist_ok=True)

# Path to dataset (using dataset_raw as requested)
images_dir = "dataset_raw/images"
labels_dir = "dataset_raw/labels"

# Get list of images
image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))

if not image_paths:
    print("No images found in dataset_raw.")
    exit()

# Pick a random image that has a corresponding label
while True:
    img_path = random.choice(image_paths)
    label_path = os.path.join(labels_dir, Path(img_path).stem + ".txt")
    if os.path.exists(label_path):
        break

print(f"Processing {img_path}")

# Load image
img = cv2.imread(img_path)
h, w, _ = img.shape

# Save Raw Image (Before)
cv2.imwrite("assets/sample_raw.jpg", img)
print("Saved assets/sample_raw.jpg")

# Draw Labels (After)
classes = ["sack", "person"]

with open(label_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, classes[cls] if cls < len(classes) else str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Save Labeled Image (After)
cv2.imwrite("assets/sample_labeled.jpg", img)
print("Saved assets/sample_labeled.jpg")
