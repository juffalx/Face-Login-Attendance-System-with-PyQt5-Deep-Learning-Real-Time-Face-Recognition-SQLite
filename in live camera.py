from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# Load models
# Use CPU for MTCNN (because of torchvision's NMS)
mtcnn = MTCNN(device='cpu', keep_all=False, image_size=240, post_process=False)

# Use GPU for face embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



# Load saved embedding for "mame"
data = torch.load("mame_embeddings.pt")
mame_embedding, mame_name = data[0]
mame_embedding = mame_embedding.to(device)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show webcam frame
    cv2.imshow("Webcam", frame)

    # Convert frame to RGB PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect face
    face = mtcnn(image)
    if face is not None:
        if face.ndimension() == 3:
            face = face.unsqueeze(0)  # [1, 3, 240, 240]
        face = face.to(device)

        with torch.no_grad():
            emb = resnet(face)[0]  # [512]

        # Compare with saved embedding (cosine similarity)
        similarity = torch.nn.functional.cosine_similarity(emb, mame_embedding, dim=0)
        print(f"Similarity: {similarity:.4f}")

        if similarity > 0.85:
            print("✅ Matched: mame")
        else:
            print("❌ Unknown face")

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
