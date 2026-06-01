# 👁️ Automated Retinal Disease Detection Using Deep Learning

> An AI-powered web application that detects retinal diseases from fundus images using Deep Learning and Computer Vision techniques.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-red)

---

# 📖 Overview

Retinal diseases are among the leading causes of vision impairment worldwide. Early diagnosis can significantly improve treatment outcomes.

This project uses a Deep Learning model based on MobileNetV2 and Transfer Learning to automatically classify retinal fundus images into multiple disease categories.

The system allows users to upload a retinal image through a web interface and instantly receive a prediction with confidence score.

---

# 🎯 Objectives

* Detect retinal diseases automatically.
* Assist doctors in preliminary diagnosis.
* Reduce manual screening effort.
* Provide fast and accurate image classification.
* Improve accessibility of retinal disease detection.

---

# 🔍 Disease Categories

The model can classify images into:

| Class                | Description                      |
| -------------------- | -------------------------------- |
| ARMD                 | Age-Related Macular Degeneration |
| Diabetic Retinopathy | Diabetes-related retinal damage  |
| Glaucoma             | Optic nerve damage               |
| Healthy              | Normal retina                    |
| Cataract             | Lens clouding condition          |
| Random Image         | Non-retinal image                |

Total Classes: **6**

---

# 🏗️ System Architecture

```text
Fundus Image
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2 Feature Extraction
      │
      ▼
Custom CNN Layers
      │
      ▼
Disease Classification
      │
      ▼
Prediction + Confidence Score
```

---

# 🧠 Deep Learning Workflow

## 1️⃣ Data Collection

Dataset Structure:

```bash
dataset/
│
├── train/
│   ├── ARMD/
│   ├── Cataract/
│   ├── Diabetic Retinopathy/
│   ├── Glaucoma/
│   ├── Healthy/
│   └── Random image/
│
└── valid/
```

---

## 2️⃣ Data Augmentation

To improve generalization, the following techniques are applied:

* Rotation
* Zoom
* Horizontal Flip
* Width Shift
* Height Shift
* Shearing

```python
ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True
)
```

---

## 3️⃣ Transfer Learning

Pre-trained Model:

```python
MobileNetV2
```

Benefits:

✅ Faster Training

✅ Higher Accuracy

✅ Reduced Overfitting

✅ Better Feature Extraction

---

## 4️⃣ Custom Classification Head

Added Layers:

```text
BatchNormalization
Conv2D (64)
MaxPooling2D
Conv2D (128)
MaxPooling2D
GlobalAveragePooling2D
Dense (512)
Dropout (0.5)
Dense (6)
```

---

## 5️⃣ Fine Tuning

The first 100 layers remain frozen.

Remaining layers are trained with:

```python
learning_rate = 1e-5
```

This improves performance while preserving learned features.

---

# 📊 Model Training

### Initial Training

```python
epochs = 40
optimizer = Adam
loss = categorical_crossentropy
```

### Fine-Tuning

```python
epochs = 10
learning_rate = 0.00001
```

### Callbacks Used

* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau
* TensorBoard

---

# 🌐 Web Application

The Flask web application provides:

### Image Upload

Users can upload:

```text
.jpg
.jpeg
.png
```

### Prediction

The uploaded image is processed and classified.

### Confidence Score

Displays model confidence percentage.

Example:

```text
Prediction:
Glaucoma

Confidence:
98.42%
```

---

# 📂 Project Structure

```bash
Automated-Retinal-Disease-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── dataset/
│   ├── train/
│   └── valid/
│
├── retinal_multiclass6_best_model.h5
├── retinal_multiclass6_finetuned_model.h5
│
└── README.md
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/retinal-disease-detection.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 📈 Performance Metrics

Evaluate using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example Results:

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 95%+  |
| Precision | High  |
| Recall    | High  |
| F1 Score  | High  |

*(Update with your actual results)*

---

# 🔬 Technologies Used

| Category          | Technology       |
| ----------------- | ---------------- |
| Programming       | Python           |
| Deep Learning     | TensorFlow/Keras |
| Computer Vision   | OpenCV           |
| Data Analysis     | NumPy            |
| Image Processing  | Pillow           |
| Web Framework     | Flask            |
| Transfer Learning | MobileNetV2      |

---

# 🎯 Applications

* Ophthalmology Screening
* Hospital Support Systems
* Telemedicine Platforms
* Rural Healthcare
* Medical Research
* AI-Assisted Diagnosis

---

# 🔮 Future Enhancements

* Multi-Disease Detection
* Severity Level Classification
* Grad-CAM Visualization
* Real-Time Camera Detection
* Cloud Deployment
* Doctor Dashboard
* Patient History Tracking

---

# 📸 Screenshots

### Home Page

Add Screenshot Here

### Image Upload

Add Screenshot Here

### Prediction Result

Add Screenshot Here

---

# 👨‍💻 Author

**Kavin S**

B.E Computer Science and Engineering

Coimbatore Institute of Technology

Project: Automated Retinal Disease Detection Using Deep Learning

---

# 📜 License

This project is intended for educational and research purposes.

MIT License

⭐ If you find this project useful, please give it a star on GitHub.
