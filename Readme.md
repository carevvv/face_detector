# Face Detection and Recognition

![Face Detection](https://media1.tenor.com/m/B8ra2i-OK9QAAAAC/face-recognition.gif)

A desktop application that detects and recognizes faces in a live camera feed. Detected faces are outlined in the video stream and matched against a gallery of known people; every detection is logged to a PostgreSQL database and can be reviewed in daily and monthly reports.

## Features

- **Face detection.** An OpenCV DNN face detector (SSD, 300×300 input) locates faces in each frame and outlines them. The detection threshold is set by `conf_threshold` (default `0.7`).
- **Face recognition.** Faces are encoded with the `face_recognition` library (dlib) and compared against reference photos in `photos/`. The matching tolerance is set by `tolerance` (default `0.6`). Comparisons run in parallel with a thread pool.
- **Detection log.** Each recognized or unknown face is saved with a timestamp and image to PostgreSQL.
- **Reports.** Daily and monthly reports of detections are available from the GUI.
- **Evaluation script.** `testing_photos.py` measures recognition accuracy on the `test_photos/` and `compare_photos/` sets.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/carevvv/face_detector.git
   cd face_detector
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the OpenCV face detector model (`opencv_face_detector.pbtxt` and `opencv_face_detector_uint8.pb`) into a `network/` directory.

4. Set the PostgreSQL connection parameters in `configuration/config.py` and create the table:

   ```bash
   python db_create.py
   ```

5. Add reference photos of known people to `photos/` (the file name is used as the person's name) and start the application:

   ```bash
   python camera.py
   ```

## Tech stack

- Python
- OpenCV (DNN face detection, video capture)
- face_recognition / dlib (face encodings)
- PyQt5 (GUI)
- PostgreSQL with the peewee ORM
