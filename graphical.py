import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceLoginApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Login System - ማመ ይግባ")
        self.setGeometry(200, 100, 800, 600)
        self.setStyleSheet("background-color: #121212; color: white; font-size: 16px;")

        # UI Elements
        self.image_label = QLabel()
        self.status_label = QLabel("📷 ካሜራ አልጀመረም")
        self.status_label.setStyleSheet("font-size: 18px; padding: 10px;")

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        for btn in [self.start_button, self.stop_button, self.quit_button]:
            btn.setStyleSheet("padding: 8px 20px; background-color: #1E88E5; color: white; border-radius: 10px;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.quit_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device.")
        self.mtcnn = MTCNN(device='cpu', image_size=240, keep_all=False, post_process=False)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known face
        self.known_embedding, self.known_name = torch.load("mame_embeddings.pt")[0]
        self.known_embedding = self.known_embedding.to(self.device)

        # Timer for video
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.status_label.setText("📸 ካሜራ ተጀመረ")
        self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.image_label.clear()
        self.status_label.setText("📷 ካሜራ ተዘጋ")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        face = self.mtcnn(pil_image)
        result_text = "❌ ያልታወቀ ፊት"

        if face is not None:
            if face.ndimension() == 3:
                face = face.unsqueeze(0)
            face = face.to(self.device)

            with torch.no_grad():
                emb = self.resnet(face)[0]
                sim = torch.nn.functional.cosine_similarity(emb, self.known_embedding, dim=0)

                if sim > 0.85:
                    result_text = "✅ ማሜ"

            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("arial.ttf", 30)  # Use Amharic font if available
            draw.text((20, 20), result_text, fill=(0, 255, 0), font=font)

        # Convert PIL back to QImage for display
        display_image = pil_image.convert("RGB")
        img_np = np.array(display_image)
        height, width, channel = img_np.shape
        bytes_per_line = channel * width
        qimg = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

        # Update status label
        self.status_label.setText(result_text)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceLoginApp()
    window.show()
    sys.exit(app.exec_())