import streamlit as st
import os
import cv2
import tempfile
import shutil
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

# Import local modules
# We need to make sure the current directory is in sys.path
import sys
sys.path.append(os.getcwd())

try:
    from auto_label import auto_label, build_sam3_image_model, Sam3Processor, get_yolo_boxes
    from video_to_frames import extract_frames
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure auto_label.py and video_to_frames.py are in the same directory.")
    st.stop()

# Page Config
st.set_page_config(page_title="Auto-Labeling Tool", layout="wide")

st.title("🤖 Auto-Labeling with SAM-3")
st.markdown("Upload videos or images, specify classes, and generate a YOLO-formatted dataset automatically.")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
classes_input = st.sidebar.text_input("Classes (comma separated)", value="sack, person")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
frame_rate = st.sidebar.number_input("Frame Extraction Rate (every N frames)", min_value=1, value=5)

classes = [c.strip() for c in classes_input.split(",") if c.strip()]

# --- Model Loading ---
@st.cache_resource
def load_model():
    print("Loading SAM-3 Model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    return model, processor

try:
    with st.spinner("Loading SAM-3 Model... (this may take a minute)"):
        model, processor = load_model()
    st.sidebar.success("Model Loaded!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# --- Main Logic ---

# Session State for persistence
if 'processed_dir' not in st.session_state:
    st.session_state.processed_dir = None
if 'zip_path' not in st.session_state:
    st.session_state.zip_path = None

tab1, tab2 = st.tabs(["Upload Video", "Upload Images"])

input_type = None
uploaded_files = []
temp_dir = "streamlit_temp"

# Cleanup temp dir on start if needed (optional, be careful)
# if os.path.exists(temp_dir):
#     shutil.rmtree(temp_dir)

with tab1:
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if video_file:
        input_type = "video"

with tab2:
    image_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if image_files:
        input_type = "images"
        uploaded_files = image_files

if st.button("Start Processing", type="primary"):
    if not input_type:
        st.error("Please upload a video or images first.")
    else:
        # Create temp directories
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. Prepare Images
        if input_type == "video":
            status_text.text("Extracting frames from video...")
            video_path = os.path.join(temp_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            extract_frames(video_path, images_dir, prefix=Path(video_file.name).stem, every_n=frame_rate)
            
        elif input_type == "images":
            status_text.text("Saving uploaded images...")
            for uploaded_file in uploaded_files:
                with open(os.path.join(images_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

        # 2. Auto-Labeling
        status_text.text("Running Auto-Labeling...")
        
        image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
        total_images = len(image_paths)
        
        if total_images == 0:
            st.error("No images found to process.")
        else:
            # We need to adapt the auto_label logic to update the progress bar
            # So we'll reimplement the loop here using the loaded model
            
            for i, img_path in enumerate(image_paths):
                try:
                    # Update progress
                    progress = (i + 1) / total_images
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {i+1}/{total_images}: {img_path.name}")
                    
                    # Load image
                    image_pil = Image.open(img_path).convert("RGB")
                    width, height = image_pil.size
                    
                    # Inference
                    inference_state = processor.set_image(image_pil)
                    
                    all_boxes = []
                    all_class_ids = []
                    
                    for class_id, class_name in enumerate(classes):
                        output = processor.set_text_prompt(state=inference_state, prompt=class_name)
                        boxes = output["boxes"]
                        scores = output["scores"]
                        
                        if boxes is not None and len(boxes) > 0:
                            for box, score in zip(boxes, scores):
                                if score > conf_threshold:
                                    all_boxes.append(box.tolist())
                                    all_class_ids.append(class_id)
                    
                    # Save Label
                    label_filename = img_path.stem + ".txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, "w") as f:
                        for box, cls in zip(all_boxes, all_class_ids):
                            yolo_box = get_yolo_boxes(box, width, height)
                            yolo_box = [max(0, min(1, x)) for x in yolo_box]
                            line = f"{cls} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                            f.write(line)
                            
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
            
            status_text.text("Processing Complete!")
            st.success(f"Successfully processed {total_images} images.")
            
            # Store result path
            st.session_state.processed_dir = temp_dir
            
            # Create ZIP
            shutil.make_archive("dataset_output", 'zip', temp_dir)
            st.session_state.zip_path = "dataset_output.zip"

# --- Results & Download ---
if st.session_state.processed_dir:
    st.divider()
    st.header("Results")
    
    # Show sample
    images_dir = os.path.join(st.session_state.processed_dir, "images")
    labels_dir = os.path.join(st.session_state.processed_dir, "labels")
    
    sample_images = list(Path(images_dir).glob("*.jpg"))[:4] # Show up to 4 samples
    
    cols = st.columns(len(sample_images))
    for idx, img_path in enumerate(sample_images):
        # Draw boxes
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        label_path = os.path.join(labels_dir, img_path.stem + ".txt")
        
        if os.path.exists(label_path):
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
                    
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255) # Green for 0, Red for 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, classes[cls] if cls < len(classes) else str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[idx].image(img_rgb, caption=img_path.name, use_container_width=True)

    # Download Button
    if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
        with open(st.session_state.zip_path, "rb") as f:
            st.download_button(
                label="Download Dataset (ZIP)",
                data=f,
                file_name="dataset_labeled.zip",
                mime="application/zip"
            )
