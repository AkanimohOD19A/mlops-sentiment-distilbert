import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from tests.test_predictor import mock_predictor

client = TestClient(app)


def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test the health check endpoint"""
    with patch("app.ml.predictor.EmotionPredictor.is_model_loaded", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}
        response_data = response.json()
        # Check that required fields exist without caring about additional fields
        assert response_data["status"] == "healthy"
        assert response_data["model_loaded"] is True


@patch("app.ml.predictor.EmotionPredictor.predict")
def test_predict_endpoint(mock_predict):
    """Test the prediction endpoint"""
    # Configure the mock to return a specific response
    mock_predict.return_value = {
        "emotion": "happy",
        "confidence": 0.92,
        "all_emotions": {"happy": 0.92, "sad": 0.05, "angry": 0.03}
    }

    # Make the request
    response = client.post(
        "/predict",
        json={"text": "I'm having a wonderful day!"}
    )

    # Verify the response
    assert response.status_code == 200
    result = response.json()
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.92
    assert "all_emotions" in result

    # Verify the mock was called with the right arguments
    mock_predict.assert_called_once_with("I'm having a wonderful day!")


@patch("app.ml.predictor.EmotionPredictor.get_labels")
def test_get_labels(mock_predictor):
    """Test the labels endpoint"""
    # Configure the mock to return specific labels
    labels = mock_predictor().get_labels()
    assert set(labels) == {"happy", "sad", "angry"}

    # Test exception when model not loaded but keep the label2id attribute
    mock_predictor.model = None  # This will make is_model_loaded() return False
    with pytest.raises(ValueError, match="Model is not loaded"):
        mock_predictor.get_labels()                                       ""}
    # mock_get_labels.return_value = ["happy", "sad", "angry"]
    #
    # # Make the request
    # response = client.get("/labels")
    #
    # # Verify the response
    # assert response.status_code == 200
    # assert response.json() == ["happy", "sad", "angry"]