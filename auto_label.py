import os
import cv2
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# SAM-3 Imports
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: SAM-3 not found. Please install it via: pip install git+https://github.com/facebookresearch/sam3.git")
    exit(1)

def get_yolo_boxes(box, img_width, img_height):
    """Convert xyxy box to YOLO format (class_id, x_center, y_center, width, height) normalized."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return [cx / img_width, cy / img_height, w / img_width, h / img_height]

def auto_label(images_dir, output_dir, classes=["sack", "person"]):
    """
    Auto-labels images using SAM-3 with text prompts.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    
    print(f"Found {len(image_paths)} images in {images_dir}")
    print("Loading SAM-3 model...")
    
    # Initialize SAM-3
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    print(f"Labeling for classes: {classes}")
    
    for img_path in tqdm(image_paths):
        label_path = os.path.join(output_dir, img_path.stem + ".txt")
        if os.path.exists(label_path):
            continue

        try:
            # Load image
            image_pil = Image.open(img_path).convert("RGB")
            width, height = image_pil.size
            
            # Set image for SAM-3
            inference_state = processor.set_image(image_pil)
            
            all_boxes = []
            all_class_ids = []
            
            for class_id, class_name in enumerate(classes):
                # Prompt with text
                output = processor.set_text_prompt(state=inference_state, prompt=class_name)
                
                # Get results
                # masks = output["masks"] # (N, H, W)
                boxes = output["boxes"] # (N, 4) in xyxy format
                scores = output["scores"]
                
                # Filter by score if needed (SAM-3 usually returns good matches for specific prompts)
                # For now, take all valid boxes
                if boxes is not None and len(boxes) > 0:
                    for box, score in zip(boxes, scores):
                        if score > 0.3: # Threshold
                            all_boxes.append(box.tolist())
                            all_class_ids.append(class_id)
            
            # Save to YOLO format
            label_path = os.path.join(output_dir, img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for box, cls in zip(all_boxes, all_class_ids):
                    yolo_box = get_yolo_boxes(box, width, height)
                    # Ensure values are within [0, 1]
                    yolo_box = [max(0, min(1, x)) for x in yolo_box]
                    
                    line = f"{cls} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                    f.write(line)
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"Labels saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output", type=str, default="dataset_raw/labels", help="Path to save labels")
    args = parser.parse_args()
    
    auto_label(args.images, args.output)
