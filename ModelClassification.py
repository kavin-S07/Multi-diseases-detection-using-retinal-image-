import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image
from datetime import datetime

# Correct path to your trained model file (.h5 or .keras)
model_path = r"C:\Users\admin\Downloads\program\program\retinal_multiclass6_finetuned_model.h5"

#  Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found at: {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Class labels (ensure this matches your training order)
class_labels = ['ARMD', 'Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Random image', 'Cataract']

def predict_image_class(image_path):
    if not os.path.exists(image_path):
        print(f" Error: Image path does not exist:\n{image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at:\n{image_path}")
        return

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Output
    print(f"\n Predicted Class: {predicted_label}")
    print(f" Confidence: {confidence:.2f}%")

    # Annotate
    annotated_img = cv2.resize(img.copy(), (400, 400))
    text = f"{predicted_label} ({confidence:.2f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x_pos = (annotated_img.shape[1] - text_size[0]) // 2
    y_pos = 40

    cv2.putText(
        annotated_img,
        text,
        (x_pos, y_pos),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"prediction_{predicted_label}_{timestamp}.jpg"
    cv2.imwrite(output_path, annotated_img)

    print(f"Annotated image saved to: {output_path}")
    Image.open(output_path).show()

# Example usage
if __name__ == "__main__":
    test_image_path = r"C:\Users\admin\Downloads\test dataset1\test dataset1\ARMD\3.png"
    predict_image_class(test_image_path)
