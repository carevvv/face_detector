from db_add import db_add_camera
import os
import sys
import cv2
from datetime import datetime, timedelta
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout,  QWidget, QVBoxLayout, QTableWidget
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtCore import Qt
from models import *
import face_recognition
import calendar
import numpy as np
import cv2
import os


faceProto = "network/opencv_face_detector.pbtxt"
faceModel = "network/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)
photos_folder = "photos"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("./TestRec.ui", self)
        self.setWindowTitle("IP Camera and Face Recognition")
        self.image_save_folder = 'ImagesAttendance'
        os.makedirs(self.image_save_folder, exist_ok=True)

        self.face_detected = False
        self.detected_persons = []
        self.face_detected_time = datetime.now() - timedelta(seconds=10)
        self.photos_encodings = {}
        self.load_and_encode_photos(photos_folder)
        self.DailyReportButton.clicked.connect(self.open_daily_report)
        self.MonthlyReportButton.clicked.connect(self.open_monthly_report)
        self.counted_faces = 0
        self.FaceBoxes = []
        self.start_video_stream()
        self.score = 0

    def start_video_stream(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.frmain)
        self.timer.start(100)


    def load_and_encode_photos(self, photos_folder):
        for photo_name in os.listdir(photos_folder):
            photo_path = os.path.join(photos_folder, photo_name)
            try:
                photo_image = face_recognition.load_image_file(photo_path)
                photo_encode = face_recognition.face_encodings(photo_image)
                if photo_encode:
                    self.photos_encodings[photo_name[:-4]] = photo_encode[0]
            except Exception as e:
                print(f"Error processing {photo_name}: {str(e)}")



    def compare_faces(self, frame, tolerance=0.6):
        frame_encodings = face_recognition.face_encodings(frame)
        if not frame_encodings:
            return []

        frame_encode = frame_encodings[0]
        recognized_faces = []
        for name, encoded_photo in self.photos_encodings.items():
            results = face_recognition.compare_faces([encoded_photo], frame_encode, tolerance)
            face_distances = face_recognition.face_distance([encoded_photo], frame_encode)
            if results[0]:
                score = 1 - face_distances[0]  
                recognized_faces.append((name, score))

        return recognized_faces


    def highlightFace(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, faceBoxes


    def frmain(self):
        hasFrame, frame = self.capture.read()
        if not hasFrame:
            return 

        resultImg, self.FaceBoxes = self.highlightFace(faceNet, frame)

        faces = self.compare_faces(frame)

        if len(faces) > 0:
            self.score = faces[0][1].round(2)
        
        self.person = [person[0] for person in faces]

        if self.FaceBoxes:
            if (datetime.now() - self.face_detected_time).total_seconds() >= 10:

                for temp in self.person:
                    db_add_camera(resultImg, temp)
                    self.detected_persons.append(temp)
                    self.counted_faces += 1

                for i in range(len(self.FaceBoxes) - len(self.person)):
                    db_add_camera(resultImg, "unknown") 
                    self.counted_faces += 1

                self.face_detected_time = datetime.now()
                self.face_detected = True
                
        self.display_image(resultImg)

        self.score = 0


    def display_image(self, img):
        qFormat = QImage.Format.Format_RGB888  
        if len(img.shape) == 3:
            r, c, channels = img.shape
            bytesPerLine = channels * c
            qImg = QImage(img.data, c, r, bytesPerLine, qFormat)
        else:
            qImg = QImage(img, img.shape[1], img.shape[0], qFormat)

        qImg = qImg.rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.imgLabel.setPixmap(pixmap)
        self.imgLabel.setScaledContents(True)

        current_date = datetime.now()
        self.Score_label.setText(str(self.score))
        self.DateLabel.setText(current_date.strftime('%d.%b.%Y'))
        self.TimeLabel.setText(current_date.strftime('%H:%M:%S'))
        if self.person:
            self.NameLabel.setText(self.person[0])
            self.HourLabel.setText(datetime.now().strftime('%H'))
            self.MinuteLabel.setText(datetime.now().strftime('%M'))
        elif self.FaceBoxes:
            self.NameLabel.setText('unknown')
            self.MinuteLabel.setText(datetime.now().strftime('%M'))
            self.HourLabel.setText(datetime.now().strftime('%H'))



    def open_daily_report(self):
        self.table_window = QtWidgets.QWidget()
        self.table_window.setWindowTitle("Daily Report")
        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)
        self.table_window.setLayout(layout)
        

        self.table_widget.setColumnCount(4)

        self.table_widget.setHorizontalHeaderLabels(["ID", "Имя", "Время", "Фото"])
        
        today_date = datetime.now().date()
        query = Camera.select().where(Camera.date >= datetime.combine(today_date, datetime.min.time()),
                                    Camera.date < datetime.combine(today_date, datetime.max.time()))
        cameras = list(query)
        
        self.table_widget.setRowCount(len(cameras))
        for index, camera in enumerate(cameras):
            self.table_widget.setItem(index, 0, QTableWidgetItem(str(camera.id)))
            self.table_widget.setItem(index, 1, QTableWidgetItem(camera.name))
            self.table_widget.setItem(index, 2, QTableWidgetItem(camera.date.strftime("%Y-%m-%d %H:%M:%S")))

            image = QImage()
            image.loadFromData(camera.picture)  
            pix = QPixmap(image).scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
            label = QtWidgets.QLabel()
            label.setPixmap(pix)
            self.table_widget.setCellWidget(index, 3, label)
        
        self.table_window.resize(800, 600)

        header = self.table_widget.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(3, 100)
        
        self.table_window.show()


    def open_monthly_report(self):
        self.table_window = QtWidgets.QWidget()
        self.table_window.setWindowTitle("Monthly Report")
        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)
        self.table_window.setLayout(layout)
        

        self.table_widget.setColumnCount(4)

        self.table_widget.setHorizontalHeaderLabels(["ID", "Имя", "Время", "Фото"])
        
        today = datetime.now().date()
        first_day, last_day = calendar.monthrange(today.year, today.month)
        start_date = datetime(today.year, today.month, 1)
        end_date = datetime(today.year, today.month, last_day)

        query = Camera.select().where(Camera.date >= start_date, Camera.date <= end_date)
        cameras = list(query)
        
        self.table_widget.setRowCount(len(cameras))
        for index, camera in enumerate(cameras):
            self.table_widget.setItem(index, 0, QTableWidgetItem(str(camera.id)))
            self.table_widget.setItem(index, 1, QTableWidgetItem(camera.name))
            self.table_widget.setItem(index, 2, QTableWidgetItem(camera.date.strftime("%Y-%m-%d %H:%M:%S")))

            image = QImage()
            image.loadFromData(camera.picture)  
            pix = QPixmap(image).scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
            label = QtWidgets.QLabel()
            label.setPixmap(pix)
            self.table_widget.setCellWidget(index, 3, label)
        
        self.table_window.resize(800, 600)

        header = self.table_widget.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(3, 100)
        
        self.table_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())