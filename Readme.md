# 🎨 Face Detection & Comparison 🌈

![Face Detection](https://media1.tenor.com/m/B8ra2i-OK9QAAAAC/face-recognition.gif)

### Face Detection and Comparison Project

This project, developed in Python, provides functionality for face detection and comparison with a database. When run, a PyQT window opens that marks your face with a red frame. If your photo is present in the database, the program recognizes and labels you. The similarity percentage can be adjusted using the `conf_threshold` variable.

---

### ⚙️ Installation

To install and run the project, follow these steps:

1. Clone the project:
    
      git clone https://github.com/carevvv/facedetector

2. Install dependencies:
    
        pip install -r requirements.txt

3. Create a database and run the database creation script:
    
        python db_create.py

4. Run the program:
    
        python camera.py

---

### 🛠 Key Features

- 😃 Face Detection: Your face is outlined with a red frame.
- 🧠 Face Comparison: If your photo is in the database, the program recognizes and labels you.
- 🎛 Similarity Adjustment: The similarity percentage can be configured with the `conf_threshold` variable.

---

### 🧑‍💻 Technologies

- 🌐 Python: The primary programming language.
- 🖼 OpenCV: Used for image processing.
- 🖥 PyQT5: For creating the graphical user interface.
- 📂 PostgreSQL: For database management.
