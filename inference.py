import argparse
import os
import cv2
import torch
import numpy as np

# Try to import YOLOX. If not found, instruct user to install it.
try:
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from yolox.data.data_augment import preproc
except ImportError:
    print("Error: YOLOX is not installed.")
    print("Please install it using: pip install yolox")
    exit(1)

# --- Configuration ---
MODEL_PATH = "Models/best_ckpt.pth"  # Path to your trained model
VIDEO_PATH = "j.mp4"                 # Path to input video
OUTPUT_PATH = "output.mp4"           # Path to save output video
CONF_THRESH = 0.3                    # Confidence threshold
NMS_THRESH = 0.45                    # NMS threshold
INPUT_SHAPE = (416, 416)             # Input size (must match training)
NUM_CLASSES = 2                      # Number of classes
CLASSES = ["sack", "person"]         # Class names (Verify order!)

def draw_boxes(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        cls_id = int(cls_ids[i])
        
        if score < conf:
            continue
            
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255) # Green for class 0, Red for class 1
        text = f"{class_names[cls_id]}: {score:.2f}"
        
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x0, y0 - text_height - 5), (x0 + text_width, y0), color, -1)
        cv2.putText(img, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return img

def main():
    # 1. Check files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # 2. Setup Experiment and Model
    print("Setting up YOLOX-Nano model...")
    # We use the standard yolox_nano exp and override params to match training
    exp = get_exp(None, "yolox_nano")
    exp.num_classes = NUM_CLASSES
    exp.depth = 0.33
    exp.width = 0.25
    exp.test_size = INPUT_SHAPE
    
    model = exp.get_model()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # 3. Load Weights
    print(f"Loading weights from {MODEL_PATH}...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print("Weights loaded successfully.")

    # 4. Process Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps}fps")
    print(f"Saving to: {OUTPUT_PATH}")
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        img, ratio = preproc(frame, INPUT_SHAPE)
        img = torch.from_numpy(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
            
        # Inference
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(
                outputs, NUM_CLASSES, CONF_THRESH,
                NMS_THRESH, class_agnostic=True
            )
            
        # Draw Detections
        output = outputs[0]
        if output is not None:
            output = output.cpu()
            bboxes = output[:, 0:4]
            # Scale boxes back to original image size
            bboxes /= ratio
            
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            
            frame = draw_boxes(frame, bboxes, scores, cls, CONF_THRESH, CLASSES)
            
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
            
    cap.release()
    out.release()
    print("Inference completed!")

if __name__ == "__main__":
    main()
