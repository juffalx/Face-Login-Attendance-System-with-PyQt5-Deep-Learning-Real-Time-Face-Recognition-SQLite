import shutil
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
#-----------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")
#---------------------------------------------------------------------------
sample_video = Path("./video/mame.mp4")

video_capture = cv2.VideoCapture(sample_video)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {frame_rate}")
    print(f"Total number of frames: {frame_count}")
    
 #-------------------------------------------------------------
    
interval = frame_rate * 0.20  # Extract every fifth frame from the video
frame_count = 0
frames_dir = Path("./extracted_image")

counter = 0

print("Start extracting individual frames...")
while True:
    # read next frame from the video_capture
    ret, frame = video_capture.read()
    if not ret:
        print("Finished!")
        break  # Break the loop if there are no more frames

    # Save frames at every 'interval' frames
    if frame_count % interval == 0:
        frame_path = frames_dir / f"frame_{counter}.jpg"
        counter += 1
        cv2.imwrite(frame_path, frame)

    frame_count += 1

video_capture.release()