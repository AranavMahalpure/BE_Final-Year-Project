import os
import pytest
from io import BytesIO

# Assuming Flask app is imported here, adjust according to your app
from app import app  # Adjust according to your Flask app module

# Create a fixture to initialize the test client
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test the index route to make sure the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200

def test_upload_missing_files(client):
    """Test uploading with missing files should return a 400 error."""
    response = client.post('/upload', data={})
    assert response.status_code == 400
    assert b'Please upload both Flair and T1ce files' in response.data

def test_upload_and_predict(client):
    """Test uploading Flair and T1ce files and ensuring prediction is returned."""
    
    # Adjust file paths to test files available for the test
    flair_path = './sample_data/sample_flair.nii'  # Replace with actual test file path
    t1ce_path = './sample_data/sample_t1ce.nii'    # Replace with actual test file path
    
    # Check if the files exist
    if not os.path.exists(flair_path) or not os.path.exists(t1ce_path):
        pytest.skip(f"Test files missing: {flair_path} or {t1ce_path}")

    with open(flair_path, 'rb') as f1, open(t1ce_path, 'rb') as f2:
        # Prepare the data as form-data
        data = {
            'flair_file': (BytesIO(f1.read()), 'sample_flair.nii'),
            't1ce_file': (BytesIO(f2.read()), 'sample_t1ce.nii')
        }
        
        # Send POST request to the /upload route
        response = client.post('/upload', data=data, content_type='multipart/form-data')
    
    # Assertions to ensure everything is working as expected
    assert response.status_code == 200
    json_data = response.get_json()
    
    # Check that the 'segmentation' key exists in the response
    assert 'segmentation' in json_data
    
    # Check that the success message is returned
    assert json_data['message'] == 'Files uploaded and processed successfully!'

    # Optionally, you can assert that the segmentation is in the expected format
    assert isinstance(json_data['segmentation'], str)  # Assuming segmentation is serialized JSON
