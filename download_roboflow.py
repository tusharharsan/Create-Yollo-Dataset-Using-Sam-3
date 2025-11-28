from roboflow import Roboflow
import os

# Download to a specific directory
os.makedirs("dataset_roboflow_2", exist_ok=True)
os.chdir("dataset_roboflow_2")

rf = Roboflow(api_key="BUitCYWkJLLFgYJ5zpnD")
project = rf.workspace("sack").project("sack-counting-ig3r3")
version = project.version(2)
dataset = version.download("yolov8")

print("Download complete.")
