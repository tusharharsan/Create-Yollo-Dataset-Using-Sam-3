import cv2
import os
import argparse
from pathlib import Path
import random

def visualize_yolo_labels(images_dir, labels_dir, output_dir, classes=["sack", "person"]):
    """
    Visualizes YOLO format labels on images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    print(f"Found {len(image_paths)} images. Visualizing...")
    
    # Random colors for classes
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
    
    count = 0
    for img_path in image_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        
        if not label_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        height, width, _ = img.shape
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Convert to pixel coordinates
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            
            # Draw box
            color = colors[class_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = classes[class_id] if class_id < len(classes) else str(class_id)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        output_path = os.path.join(output_dir, img_path.name)
        cv2.imwrite(output_path, img)
        count += 1
        
    print(f"Saved {count} visualized images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="dataset_raw/images", help="Path to images")
    parser.add_argument("--labels", type=str, default="dataset_raw/labels", help="Path to labels")
    parser.add_argument("--output", type=str, default="dataset_raw/visualized", help="Path to save output")
    args = parser.parse_args()
    
    visualize_yolo_labels(args.images, args.labels, args.output)
