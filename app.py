from flask import Flask, render_template, request, jsonify
import os
import nibabel as nib
import numpy as np
import cv2
import tensorflow as tf
import json
from tensorflow.keras.models import load_model

app = Flask(__name__)

SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}

# Priority labels for low grade and high grade
LOW_GRADE_CLASSES = ['EDEMA']
HIGH_GRADE_CLASSES = ['NECROTIC/CORE', 'ENHANCING']
def determine_tumor_grade(serialized_prediction):
    # Convert the serialized prediction back into a list of labels
    prediction_data = json.loads(serialized_prediction)  # Deserialize the prediction
    
    # Flatten the prediction data to make it easier to check each value
    flattened_data = [item for sublist in prediction_data for item in sublist]
    
    # Check for high grade tumor (NECROTIC/CORE or ENHANCING)
    if any(label in HIGH_GRADE_CLASSES for label in flattened_data):
        return "high grade"
    
    # Check for low grade tumor (NOT tumor, EDEMA)
    elif any(label in LOW_GRADE_CLASSES for label in flattened_data):
        return "low grade"
    
    # If neither is detected
    return "Tumor Not Detected" 

def convert_to_serializable(prediction):
    """
    Convert the mapped prediction array to a JSON-serializable format.
    """
    return json.dumps(prediction.tolist())

def map_prediction_to_labels(prediction):
    """
    Map the predicted 2D array of class indices to their corresponding labels.
    """
    label_map = np.vectorize(SEGMENT_CLASSES.get)(prediction)
    return label_map

# Define upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "3D_MRI_Brain_tumor_segmentation(35).h5"
model = load_model(MODEL_PATH)
# Define image size expected by the model
IMG_SIZE = 128
def preprocess_image(image_file, slice_index=None):
    # Load the NIfTI file
    img = nib.load(image_file).get_fdata()
    
    # Select a specific slice if needed
    if slice_index is not None:
        img = img[:, :, slice_index]
    
    # Resize the image to (IMG_SIZE, IMG_SIZE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize the image
    img_resized = img_resized / np.max(img_resized)
    
    return img_resized
def predict(image_paths, slice_index):
    X = np.zeros((1, IMG_SIZE, IMG_SIZE, 2))
    
    # Process specific slices from the images
    X[0, :, :, 0] = preprocess_image(image_paths[0], slice_index)
    X[0, :, :, 1] = preprocess_image(image_paths[1], slice_index)
    
    # Make prediction
    pred = model.predict(X)
    
    return np.argmax(pred[0], axis=-1)

def get_classification(pred):
    # Get the class with the highest probability for each pixel
    class_predictions = np.argmax(pred, axis=-1)
    return class_predictions
import matplotlib.pyplot as plt

def visualize_prediction(prediction):
    plt.imshow(prediction, cmap='nipy_spectral')  # 'nipy_spectral' gives distinct colors for different classes
    plt.title('Predicted Segmentation')
    plt.colorbar()
    plt.show()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'flair_file' not in request.files or 't1ce_file' not in request.files:
        return jsonify({'error': 'Please upload both Flair and T1ce files!'}), 400
    
    flair_file = request.files['flair_file']
    t1ce_file = request.files['t1ce_file']

    if flair_file.filename == '' or t1ce_file.filename == '':
        return jsonify({'error': 'One or both files are missing!'}), 400

    # Save files to the 'uploads' folder
    flair_path = os.path.join(app.config['UPLOAD_FOLDER'], flair_file.filename)
    t1ce_path = os.path.join(app.config['UPLOAD_FOLDER'], t1ce_file.filename)

    flair_file.save(flair_path)
    t1ce_file.save(t1ce_path)
    image_paths = [
    flair_path,
    t1ce_path
]
    slice_index = 75  # Example slice index
    prediction = predict(image_paths, slice_index)
    # Map the predicted class indices to their corresponding class names
    mapped_prediction = np.vectorize(SEGMENT_CLASSES.get)(prediction)
    # Convert mapped prediction to a JSON-serializable format
    serializable_prediction = convert_to_serializable(mapped_prediction)
    # Determine tumor grade
    print(serializable_prediction)
    tumor_grade = determine_tumor_grade(serializable_prediction)
        # Render the result.html template with tumor grade
    return render_template('result.html', tumor_grade=tumor_grade)

if __name__ == '__main__':
    app.run(debug=True)
