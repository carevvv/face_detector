import face_recognition
import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming faceProto and faceModel have been correctly loaded along with faceNet
faceProto = "network/opencv_face_detector.pbtxt"
faceModel = "network/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

test_photos_folder = "test_photos"
compare_photos_folder = "compare_photos"

def load_and_encode_photos(photo_folder):
    photos_encodings = {}
    for photo_name in os.listdir(photo_folder):
        photo_path = os.path.join(photo_folder, photo_name)
        try:
            photo_image = face_recognition.load_image_file(photo_path)
            photo_encode = face_recognition.face_encodings(photo_image)
            if photo_encode:
                photos_encodings[photo_name[:-4]] = photo_encode[0]
        except Exception as e:
            print(f"Error processing {photo_name}: {str(e)}")
    return photos_encodings


photos_encodings = load_and_encode_photos(test_photos_folder)

def compare_faces(image_array, tolerance=0.6):
    frame_encodings = face_recognition.face_encodings(image_array)
    if not frame_encodings:
        return []
    
    frame_encode = frame_encodings[0]
    with ThreadPoolExecutor() as executor:
        tasks = {executor.submit(face_recognition.compare_faces, [frame_encode], enc, tolerance=tolerance): name for name, enc in photos_encodings.items()}
        similarity = [name for future in as_completed(tasks) for name, result in [(tasks[future], future.result()[0])] if result]
        return similarity

def main():
    for filename in os.listdir(compare_photos_folder):
        frame_path = os.path.join(compare_photos_folder, filename)
        frame = face_recognition.load_image_file(frame_path)
        person = compare_faces(frame)
        if person:
            print(person[0])

if __name__ == "__main__":
    main()
