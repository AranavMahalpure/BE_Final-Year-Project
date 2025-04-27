import numpy as np
from app import map_prediction_to_labels, convert_to_serializable

def test_map_prediction_to_labels():
    arr = np.array([[0, 1], [2, 3]])
    result = map_prediction_to_labels(arr)
    expected = np.array([['NOT tumor', 'NECROTIC/CORE'], ['EDEMA', 'ENHANCING']])
    assert np.array_equal(result, expected)

def test_convert_to_serializable():
    arr = np.array([[1, 2], [3, 0]])
    json_str = convert_to_serializable(arr)
    assert isinstance(json_str, str)
