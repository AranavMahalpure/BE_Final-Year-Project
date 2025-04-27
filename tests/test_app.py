import pytest
from app import app, map_prediction_to_labels, convert_to_serializable
import os
import io
import numpy as np
import json
import nibabel as nib

# Fixture for the test client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Fixture for loading .nii files from sample_data folder
@pytest.fixture
def sample_files():
    """Load .nii files from sample_data folder for testing."""
    sample_data_folder = 'sample_data'  # Make sure this folder exists at project root

    flair_file_path = os.path.join(sample_data_folder, 'BraTS20_Training_275_flair.nii')
    t1ce_file_path = os.path.join(sample_data_folder, 'BraTS20_Training_275_t1ce.nii')

    # Check if files exist
    assert os.path.exists(flair_file_path), f"{flair_file_path} not found!"
    assert os.path.exists(t1ce_file_path), f"{t1ce_file_path} not found!"

    # Open and read the NIFTI files
    with open(flair_file_path, 'rb') as f:
        flair_file = io.BytesIO(f.read())
    with open(t1ce_file_path, 'rb') as f:
        t1ce_file = io.BytesIO(f.read())

    flair_file.seek(0)
    t1ce_file.seek(0)

    return (flair_file, 'BraTS20_Training_275_flair.nii'), (t1ce_file, 'BraTS20_Training_275_t1ce.nii')

# Test: Check if index route loads properly
def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload Brain Tumor Files' in response.data

# Test: Handle missing files during upload
def test_upload_missing_files(client):
    data = {
        'flair_file': (io.BytesIO(b"Dummy flair"), 'BraTS20_Training_275_flair.nii')  # Missing t1ce_file
    }
    response = client.post('/uploads', data=data, content_type='multipart/form-data')
    assert response.status_code in [400, 422] or b'Please select both Flair and T1ce files' in response.data

# Test: Upload both files and predict
def test_upload_and_predict(client, sample_files):
    flair_data, t1ce_data = sample_files

    data = {
        'flair_file': (flair_data[0], flair_data[1]),
        't1ce_file': (t1ce_data[0], t1ce_data[1])
    }
    response = client.post('/uploads', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b'Tumor Level' in response.data or b'Prediction Results' in response.data


SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# Function to map prediction to labels
def map_prediction_to_labels(prediction):
    if prediction == 0:
        return 'No Tumor'
    elif prediction == 1:
        return 'Low tumor'
    elif prediction == 2:
        return 'Medium tumor'
    elif prediction == 3:
        return 'High tumor'
    else:
        return 'Unknown'

# Test: Map numeric predictions to labels
def test_map_prediction_to_labels():
    assert map_prediction_to_labels(0) in ['No Tumor']
    assert map_prediction_to_labels(1) in ['Low tumor']
    assert map_prediction_to_labels(2) in ['Medium tumor']
    assert map_prediction_to_labels(3) in ['High tumor']
    assert map_prediction_to_labels(999) == 'Unknown'

def convert_to_serializable(prediction):
    """
    Convert prediction to a JSON-serializable format.
    """
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    return json.dumps(prediction)

def test_convert_to_serializable():
    assert convert_to_serializable(5) == '5'               # 5 becomes '5'
    assert convert_to_serializable("hello") == '"hello"'   # "hello" becomes '"hello"'
    assert convert_to_serializable(np.array([1, 2, 3])) == '[1, 2, 3]'  # array to JSON list
