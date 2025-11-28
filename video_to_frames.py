import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, prefix="frame", every_n=1):
    """
    Extracts frames from a video file and saves them with unique names.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save extracted frames.
        prefix (str): Prefix for the frame filenames (e.g., video name).
        every_n (int): Save every Nth frame (to reduce redundancy).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    video_name = Path(video_path).stem
    if prefix == "frame":
        prefix = video_name

    print(f"Extracting frames from {video_path}...")
    print(f"Saving to {output_dir} with prefix '{prefix}'...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n == 0:
            # Create unique filename: prefix_frameID.jpg
            # Using 6 digits for frame ID to ensure correct sorting
            filename = f"{prefix}_{frame_count:06d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames from {frame_count} total frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for dataset creation.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default="dataset_raw/images", help="Directory to save frames.")
    parser.add_argument("--prefix", type=str, default="frame", help="Prefix for filenames (default: video filename).")
    parser.add_argument("--every_n", type=int, default=5, help="Save every Nth frame (default: 5).")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir, args.prefix, args.every_n)
