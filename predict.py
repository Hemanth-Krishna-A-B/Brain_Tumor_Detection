import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


model = tf.keras.models.load_model('brain_tumor_classifier.h5')
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError("The image could not be loaded.")

   
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(image_path):
    try:
        
        processed_image = preprocess_image(image_path, 200)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        print(f"Predicted Class: {categories[predicted_class]} (Confidence: {confidence:.2f}%)")

        image = cv2.imread(image_path, 0)
        plt.imshow(image, cmap='gray')
        plt.title(f"Prediction: {categories[predicted_class]}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

image_path = 'pit001.jpg'
predict_image(image_path)
