import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def merge_datasets(source_dir, dest_dir, split="train"):
    """
    Merges source dataset into destination dataset.
    Assumes YOLO format:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    # Standard YOLOv8: images/train
    source_images = Path(source_dir) / "images" / split
    source_labels = Path(source_dir) / "labels" / split
    
    # Handle Roboflow structure: train/images
    if not source_images.exists() and (Path(source_dir) / split / "images").exists():
        print(f"Note: Found Roboflow structure in {Path(source_dir) / split / 'images'}")
        source_images = Path(source_dir) / split / "images"
        source_labels = Path(source_dir) / split / "labels"

    # Handle flat directory structure (like dataset_raw)
    elif not source_images.exists() and (Path(source_dir) / "images").exists():
        print(f"Note: {source_images} not found. Using flat structure in {Path(source_dir) / 'images'}")
        source_images = Path(source_dir) / "images"
        source_labels = Path(source_dir) / "labels"
    
    dest_images = Path(dest_dir) / "images" / split
    dest_labels = Path(dest_dir) / "labels" / split
    
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)
    
    # Get all images in source
    images = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
    print(f"Merging {len(images)} images from {source_images} to {dest_images}...")
    
    for img_path in tqdm(images):
        # Check for label
        label_path = source_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"Warning: No label found for {img_path.name}, skipping.")
            continue
            
        # Check for collision in destination
        dest_img_path = dest_images / img_path.name
        dest_label_path = dest_labels / label_path.name
        
        if dest_img_path.exists():
            print(f"Warning: {img_path.name} already exists in destination. Renaming...")
            # Try finding a unique name
            counter = 1
            while True:
                new_name = f"{img_path.stem}_{counter}{img_path.suffix}"
                dest_img_path = dest_images / new_name
                if not dest_img_path.exists():
                    break
                counter += 1
            
            dest_label_path = dest_labels / (Path(new_name).stem + ".txt")
            
        # Copy files
        shutil.copy2(img_path, dest_img_path)
        shutil.copy2(label_path, dest_label_path)
        
    print("Merge completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Source dataset root")
    parser.add_argument("--dest", type=str, required=True, help="Destination dataset root")
    parser.add_argument("--split", type=str, default="train", help="Split to merge (train/val)")
    args = parser.parse_args()
    
    merge_datasets(args.source, args.dest, args.split)
