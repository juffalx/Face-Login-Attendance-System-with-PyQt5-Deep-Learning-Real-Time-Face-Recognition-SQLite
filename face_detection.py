import os
import shutil
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN ,InceptionResnetV1
from PIL import Image
from torchvision.utils import make_grid

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")


mtcnn = MTCNN(device = device, keep_all = True, min_face_size = 40, image_size=240, post_process=False)

#------------------------------------------------------------------
current_dictionary = Path.cwd()
input_dir = Path("/extracted_image")
output_dir = current_dictionary / "images_dir/mame"

jpg_files = os.listdir(input_dir)
#{file.name: str(file) for file in input_dir.glob("*.jpg")}
print(jpg_files)
#------------------------------------------------------------------
#------------------------------------------------------------------

resnet = InceptionResnetV1(pretrained = "vggface2")
# embedding_data = torch.load("mame_embeddings.pt")
# resnet = resnet.eval()
#------------------------------------------------------------------
#------------------------------------------------------------------

def locate_faces(image):
    cropped_images, probs = mtcnn(image, return_prob = True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))
   

emb_data = []
    
#------------------------------------------------------------------
for img in jpg_files:
    box,probs,faces = locate_faces(img)
    if faces is not None:
        if probs >= 0.9:
            if face.ndimension() == 3:
                face = face.unsqueeze(0)
                
            emb = resnet(face)
            emb_data.append(emb)
        else:
            print("embading is under 0.9 and ", probs)
    
#------------------------------------------------------------------
#------------------------------------------------------------------
embading_mame = torch.stack(emb_data)
avg_embedding_mame = torch.mean(emb_data, dim=0)

embeddings_to_save = [(avg_embedding_mame, "mame")]
torch.save(embeddings_to_save, "mame_embeddings.pt")

#------------------------------------------------------------------
#------------------------------------------------------------------

#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
