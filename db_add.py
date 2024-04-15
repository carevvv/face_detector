from models import *
from peewee import *
import cv2
from datetime import datetime


def db_add_camera(frame, name):

    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    with connection.atomic():
        Camera.create(date=datetime.now(), name=name, picture=image_bytes)
