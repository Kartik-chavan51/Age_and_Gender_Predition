import tensorflow as tf
import numpy as np
import cv2

base = r'D:\DS projects\Ageand Gender prediciton\age+gender+prediction+Project+Code\Project Code\app'

# Print TensorFlow version for debugging
print("\n\nTensorFlow Version:", tf.__version__, "\n\n")

# Load models
model_gender = tf.keras.models.load_model(f'{base}/my_model.keras')
model_age = tf.keras.models.load_model(f'{base}/my_model1.keras')

def image_pre(path):
    """Preprocesses the image for model prediction."""
    print(f"Processing image: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error: Image at {path} could not be loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = 255 - img  # Invert colors (negative)
    img = cv2.resize(img, (128, 128))  # Resize to match model input

    # Normalize and reshape for model input
    data = np.array(img).reshape((-1, 128, 128, 1)) / 255.0
    return data

def predict(data):
    """Predicts age and gender from preprocessed image data."""
    print("Predicting age and gender...")

    pred_age = model_age.predict(data)[0][0]  # Extract scalar value
    pred_gen = model_gender.predict(data)[0][0]  # Ensure it's a single value

    age = int(round(pred_age))  # Convert to integer age
    gender = int(round(pred_gen))  # Convert to 0 (Female) or 1 (Male)

    print(f"Predicted Age: {age}, Predicted Gender: {'Male' if gender == 1 else 'Female'}")
    
    return age, gender
