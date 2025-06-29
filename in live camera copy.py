from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFont, ImageDraw
import torch
import cv2
import cvzone
from cvzone.Utils import putTextRect
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# Models
mtcnn = MTCNN(device='cpu', keep_all=False, image_size=240, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embedding
data = torch.load("mame_embeddings.pt")
mame_embedding, mame_name = data[0]
mame_embedding = mame_embedding.to(device)

# Face detector for bounding box
detector = FaceDetector()

# Webcam
cap = cv2.VideoCapture(0)

# Load Amharic font (make sure the .ttf file is in your project folder)
amharic_font_path = "NotoSansEthiopic-Regular.ttf"  # Example font for Amharic
font = ImageFont.truetype(amharic_font_path, 32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(rgb_image)

    label = "ፊት አልተገኘም"  # Default in Amharic: "No face found"
    match_found = False

    if face is not None:
        if face.ndimension() == 3:
            face = face.unsqueeze(0)
        face = face.to(device)

        with torch.no_grad():
            emb = resnet(face)[0]

        similarity = torch.nn.functional.cosine_similarity(emb, mame_embedding, dim=0)
        print(f"Similarity: {similarity:.4f}")

        if similarity > 0.85:
            label = "✅ ማሜ"  # Amharic: “Mame Found”
            match_found = True
        else:
            label = "❌ አልታወቀም"  # Unknown face

    # Detect face box for drawing
    img, bboxs = detector.findFaces(frame, draw=False)
    if bboxs:
        x, y, w, h = bboxs[0]['bbox']
        cvzone.cornerRect(img, (x, y, w, h), l=15, t=3, colorR=(255, 0, 255))

        # Render Amharic label using PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y - 35), label, font=font, fill=(0, 255, 0))

        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Face Login (Amharic)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
