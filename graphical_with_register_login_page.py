# face_login_register_app.py

import sys
import cv2
import torch
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(200, 100, 800, 600)

        # Layouts
        self.layout = QVBoxLayout()
        self.cam_label = QLabel("Webcam will appear here")
        self.status_label = QLabel("Status: Waiting")
        self.name_input = QLineEdit("Name")
        self.email_input = QLineEdit("Email")
        self.password_input = QLineEdit("Password")
        self.password_input.setEchoMode(QLineEdit.Password)

        self.btn_start = QPushButton("Start Camera")
        self.btn_register = QPushButton("Register User")
        self.btn_login = QPushButton("Login/Mark Attendance")

        # Layout add
        self.layout.addWidget(self.cam_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.email_input)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.btn_start)
        self.layout.addWidget(self.btn_register)
        self.layout.addWidget(self.btn_login)
        self.setLayout(self.layout)

        # Button events
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_register.clicked.connect(self.register_user)
        self.btn_login.clicked.connect(self.login_user)

        # Face detection & recognition
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(device='cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frames = []
        self.recording = False

        # Database
        self.conn = sqlite3.connect('users.db')
        self.create_db()

    def create_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            password TEXT,
            embedding BLOB
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            timestamp TEXT
        )''')
        self.conn.commit()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.frames.clear()
        self.timer.start(30)
        self.status_label.setText("Camera Started")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frames.append(frame)
        if len(self.frames) > 300:
            self.frames.pop(0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(img))

    def register_user(self):
        name = self.name_input.text()
        email = self.email_input.text()
        password = self.password_input.text()

        if not all([name, email, password]):
            QMessageBox.warning(self, "Missing", "Please fill all fields")
            return

        self.status_label.setText("Recording video for embedding...")
        self.recording = True
        QApplication.processEvents()
        video_frames = self.frames[-300:]  # last ~10 seconds

        faces = []
        for f in video_frames:
            face = self.mtcnn(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            if face is not None:
                if face.ndimension() == 3:
                    face = face.unsqueeze(0)
                faces.append(face)

        if not faces:
            QMessageBox.warning(self, "Error", "No face detected")
            return

        face_stack = torch.cat(faces).to(self.device)
        with torch.no_grad():
            embeddings = self.model(face_stack)
            mean_embedding = embeddings.mean(dim=0).cpu()

        c = self.conn.cursor()
        c.execute("INSERT INTO users (name, email, password, embedding) VALUES (?, ?, ?, ?)",
                  (name, email, password, mean_embedding.numpy().tobytes()))
        self.conn.commit()
        self.status_label.setText("User registered successfully")

    def login_user(self):
        self.status_label.setText("Checking face for login...")
        QApplication.processEvents()
        video_frames = self.frames[-100:]  # last few seconds

        found = False
        for f in video_frames:
            face = self.mtcnn(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            if face is None:
                continue
            if face.ndimension() == 3:
                face = face.unsqueeze(0)
            face = face.to(self.device)

            with torch.no_grad():
                emb = self.model(face)[0].cpu()

            c = self.conn.cursor()
            for row in c.execute("SELECT id, name, embedding FROM users"):
                uid, uname, emb_bytes = row
                db_emb = torch.tensor(np.frombuffer(emb_bytes, dtype=np.float32))
                sim = torch.nn.functional.cosine_similarity(emb, db_emb, dim=0)
                if sim > 0.85:
                    c.execute("INSERT INTO attendance (user_id, timestamp) VALUES (?, ?)",
                              (uid, datetime.now().isoformat()))
                    self.conn.commit()
                    self.status_label.setText(f"✅ Welcome {uname}")
                    found = True
                    break
            if found:
                break

        if not found:
            self.status_label.setText("❌ Unknown Face")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.conn.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FaceApp()
    win.show()
    sys.exit(app.exec_())
