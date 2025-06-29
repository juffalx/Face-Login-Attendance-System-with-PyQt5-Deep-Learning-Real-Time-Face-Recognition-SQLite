from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# Models
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=40, image_size=240, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Directories
input_dir = Path("./extracted_image")  # Make sure this exists
jpg_files = list(input_dir.glob("*.jpg"))

# Embedding collection
emb_data = []

# Process each image
for img_path in jpg_files:
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to open {img_path.name}: {e}")
        continue

    boxes, probs = mtcnn.detect(image)
    faces = mtcnn(image)

    if faces is not None and probs is not None:
        for prob, face in zip(probs, faces):
            if prob >= 0.9:
                if face.ndimension() == 3:
                    face = face.unsqueeze(0)  # Add batch dimension
                face = face.to(device)
                with torch.no_grad():
                    emb = resnet(face)
                emb_data.append(emb)
            else:
                print(f"⛔ Skipped low prob face ({prob:.2f}) in {img_path.name}")
    else:
        print(f"😕 No face found in {img_path.name}")

# Save embeddings
if emb_data:
    emb_stack = torch.cat(emb_data)
    avg_embedding = torch.mean(emb_stack, dim=0)

    embeddings_to_save = [(avg_embedding, "mame")]
    torch.save(embeddings_to_save, "mame_embeddings.pt")
    print("✅ Saved embeddings to mame_embeddings.pt")
else:
    print("⚠️ No valid face embeddings were collected.")