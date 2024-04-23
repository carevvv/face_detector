import cv2
import face_recognition
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import numpy as np

# Face Net model
faceProto = "network/opencv_face_detector.pbtxt"
faceModel = "network/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Folder paths
test_photos_folder = "test_photos"
compare_photos_folder = "compare_photos"

def load_and_encode_photos(photo_folder):
    photos_encodings = {}
    for photo_name in os.listdir(photo_folder):
        photo_path = os.path.join(photo_folder, photo_name)
        photo_image = face_recognition.load_image_file(photo_path)
        photo_encode = face_recognition.face_encodings(photo_image)
        if photo_encode:
            photos_encodings[photo_name[:-4]] = photo_encode[0]
    return photos_encodings

photos_encodings = load_and_encode_photos(test_photos_folder)

def highlightFace(net, frame, conf_threshold=0.7):
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

def compare_faces(image):
    frame_encodings = face_recognition.face_encodings(image)
    if frame_encodings:
        frame_encode = frame_encodings[0]
        with ThreadPoolExecutor() as executor:
            tasks = {executor.submit(face_recognition.compare_faces, [frame_encode], enc): name for name, enc in photos_encodings.items()}
            for future in as_completed(tasks):
                if True in future.result():
                    person_name = tasks[future]
                    return person_name
    return None

def resize_image(image, target_height, interpolation=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    scaling_factor = target_height / float(h)
    new_width = int(w * scaling_factor)
    return cv2.resize(image, (new_width, target_height), interpolation=interpolation)

def main():
    compare_images = os.listdir(compare_photos_folder)
    count_true = 0
    count = 0 
    for filename in compare_images:
        count += 1 
        comparison_image_path = os.path.join(compare_photos_folder, filename)
        comparison_image = face_recognition.load_image_file(comparison_image_path)
        comparison_image_cv2 = cv2.imread(comparison_image_path)
        person_identified = compare_faces(comparison_image)
        if person_identified.strip() == filename[:-4].strip():
            count_true += 1
        title = "Person not detected"
        matched_person_image = None
        if person_identified:
            title = person_identified
            person_image_path = os.path.join(test_photos_folder, person_identified + ".jpg")
            matched_person_image = cv2.imread(person_image_path)
        
        marked_image, face_boxes = highlightFace(faceNet, comparison_image_cv2)
        
        if matched_person_image is None:
            matched_person_image = np.zeros_like(marked_image)

        target_height = max(marked_image.shape[0], matched_person_image.shape[0])
        
        marked_image = resize_image(marked_image, target_height)
        matched_person_image = resize_image(matched_person_image, target_height)

        combined_image = np.hstack((marked_image, matched_person_image))

        # Масштабируем изображение для отображения
        display_image = resize_image(combined_image, 500)  # или любая другая высота, подходящая для вашего дисплея
        cv2.putText(display_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Results', display_image)
        
        cv2.waitKey(1000) 
        cv2.destroyAllWindows()

    accuracy = count_true / count * 100
    print("Accuracy: {:.2f}%".format(accuracy))
    
    # Отображение окна с точностью
    accuracy_image = np.zeros((100, 400, 3), dtype="uint8")
    cv2.putText(accuracy_image, f"Accuracy: {accuracy:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Accuracy", accuracy_image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()