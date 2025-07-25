from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# ✅ FIXED: Use raw string (r"") or double backslashes
model_path = r"C:\Users\admin\OneDrive\Documents\3rd year mini project\program\program\retinal_multiclass6_finetuned_model.h5"
model = tf.keras.models.load_model(model_path)

# Folder for uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels = ['ARMD', 'Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Random image', 'Cataract']

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    index = np.argmax(prediction)
    label = class_labels[index]
    confidence = float(prediction[index]) * 100
    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, conf = predict_image(filepath)
            prediction = label
            confidence = f"{conf:.2f}"
            uploaded_image = filename

    return render_template('index.html', prediction=prediction, confidence=confidence, image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)
