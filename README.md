# 🤟 Hand Gesture to Speech System

A real-time **Hand Gesture to Text and Speech Recognition System** built using **Python, MediaPipe, OpenCV, Scikit-learn, and Text-to-Speech**.

The project recognizes predefined hand gestures through a webcam, converts them into text, combines multiple gestures into meaningful sentences, and produces voice output.

For example:

`I → NEED → WATER → "I need water" → 🔊 Speech Output`

---

## 📌 Project Overview

Communication can be difficult for people with speech impairments. This project aims to provide a simple assistive communication system where hand gestures are recognized using computer vision and machine learning.

The detected gestures are converted into readable text and then spoken aloud using a Text-to-Speech engine.

The system currently supports gesture-based phrases such as:

- I
- Need
- Water
- Love
- You

Example sentences:

- **I need water**
- **I love you**

---

## ✨ Features

- 🎥 Real-time hand gesture recognition using webcam
- ✋ Hand landmark detection using MediaPipe
- 🤖 Machine Learning based gesture classification
- 📝 Gesture-to-text conversion
- 🔊 Text-to-Speech output
- 🧠 Multiple gesture combination to form sentences
- ⚡ Real-time prediction
- 📊 Custom gesture dataset collection
- 💾 Trained model saved using Joblib
- 🎯 Confidence threshold for reliable predictions
- 🔁 Stable-frame detection to reduce incorrect predictions

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam capture and image processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Machine Learning model training |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Joblib | Saving and loading trained ML models |
| Windows SAPI | Text-to-Speech output |
| PyWin32 | Accessing Windows speech services |

---

## 🧠 How It Works

The project follows the pipeline:

### 1. Webcam Input

The webcam continuously captures video frames using OpenCV.

### 2. Hand Detection

MediaPipe Hand Landmarker detects the hand and extracts hand landmarks.

Each hand contains **21 landmark points**.

### 3. Feature Extraction

The landmark coordinates are converted into numerical feature vectors.

These values represent the position of different parts of the hand.

### 4. Normalization

The landmark coordinates are normalized so that gesture recognition is less affected by:

- Hand position
- Distance from camera
- Slight movement variations

### 5. Gesture Classification

The trained Scikit-learn machine learning model receives the feature vector and predicts the gesture.

Example:

`Hand Landmarks → ML Model → "WATER"`

### 6. Sentence Formation

Recognized gestures are stored sequentially.

Example:

`I`

↓

`NEED`

↓

`WATER`

↓

`I NEED WATER`

### 7. Text-to-Speech

The final sentence is passed to the Windows Text-to-Speech engine.

Output:

🔊 **"I need water"**

---

## 📂 Project Structure

```text
hand-gesture-to-speech/
│
├── Utilis.py
├── data_collection.py
├── train_model.py
├── gesture_to_speech.py
│
├── gesture_data.csv
├── gesture_model.joblib
├── gesture_labels.joblib
├── hand_landmarker.task
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

**Utilis.py**

Contains common utility functions such as:

- Creating MediaPipe Hand Landmarker
- Drawing hand landmarks
- Converting landmarks into feature vectors
- Converting OpenCV frames to MediaPipe image format

**data_collection.py**

Used to collect hand gesture samples through the webcam.

The collected landmark values are stored in:

```text
gesture_data.csv
```

**train_model.py**

Loads the collected dataset and trains the machine learning model using Scikit-learn.

The trained files are saved as:

```text
gesture_model.joblib
gesture_labels.joblib
```

**gesture_to_speech.py**

Main application used for:

- Real-time gesture recognition
- Gesture prediction
- Sentence formation
- Text-to-Speech output

---

## 📊 Model Performance

During initial training with gestures such as:

- I
- Need
- Water

The model achieved approximately:

```text
Accuracy: 99%
```

Example classification results:

| Gesture | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| I | 1.00 | 0.98 | 0.99 |
| Need | 1.00 | 0.99 | 0.99 |
| Water | 0.97 | 1.00 | 0.99 |

Overall model accuracy:

**99%**

> Performance may vary depending on lighting conditions, hand position, camera quality, dataset size, and user variation.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hand-gesture-to-speech.git
```

Move into the project folder:

```bash
cd hand-gesture-to-speech
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install Required Libraries

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pywin32
```

Or install using:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Collection

Run:

```bash
python data_collection.py
```

Use the configured controls to select a gesture and record training samples.

Example controls:

```text
0-9   → Select gesture
SPACE → Start/Stop recording
Q     → Quit
```

The recorded gesture landmarks will be saved in:

```text
gesture_data.csv
```

---

## 🧪 Train the Model

After collecting enough samples, run:

```bash
python train_model.py
```

The program trains the machine learning classifier and saves the trained model.

Generated files:

```text
gesture_model.joblib
gesture_labels.joblib
```

---

## ▶️ Run the Application

Run:

```bash
python gesture_to_speech.py
```

The webcam will open and start detecting hand gestures.

Example:

```text
Gesture 1: I

Gesture 2: NEED

Gesture 3: WATER

Sentence: I NEED WATER
```

The computer will then speak:

🔊 **"I need water"**

---

## 🎯 Prediction Stability

To avoid accidental predictions, the system uses a confidence threshold and stable-frame detection.

Example configuration:

```python
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES_REQUIRED = 4
```

A gesture must remain stable for multiple frames before being accepted.

This helps reduce false detections.

---

## 🔮 Future Improvements

Future versions of this project can include:

- Larger gesture vocabulary
- Complete sign language recognition
- Dynamic gesture recognition
- LSTM/CNN based gesture recognition
- Two-hand gesture recognition
- Multiple-user training
- Mobile application integration
- Android support
- Web-based gesture recognition
- Cloud-based speech processing
- Multilingual speech output
- Automatic sentence generation
- Emergency communication mode
- Integration with smart assistive devices

---

## 🌍 Applications

This project can be useful in:

- Assistive communication systems
- Speech-impaired communication
- Human-computer interaction
- Smart healthcare systems
- Sign language interpretation
- Emergency communication
- Accessibility technology
- AI-powered assistive devices

---

## 🎓 Learning Outcomes

Through this project, concepts such as the following are explored:

- Computer Vision
- Machine Learning
- Hand Landmark Detection
- Feature Extraction
- Data Collection
- Data Preprocessing
- Model Training
- Classification
- Real-time Prediction
- Text-to-Speech
- Human-Computer Interaction

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

You can:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request

---

## 📜 License

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

Developed as an **AI/ML and Computer Vision project** focused on creating an accessible communication system using hand gesture recognition.


## ⭐ Support

If you find this project useful, consider giving the repository a **⭐ Star**.

It helps support the project and future improvements.
