import pytest
import os
import json
import torch
from app.ml.predictor import EmotionPredictor
from unittest.mock import patch, MagicMock

# Mock data for testing
@pytest.fixture
def mock_predictor():
    """Create a predictor with mocked components"""
    with patch("app.ml.predictor.AutoModelForSequenceClassification") as mock_model_cls, \
         patch("app.ml.predictor.AutoTokenizer") as mock_tokenizer_cls:

        # Configure mocks
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock model config
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "happy", 1: "sad", 2: "angry"}
        mock_model.config.label2id = {"happy": 0, "sad": 1, "angry": 2}

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Create outputs for the model
        outputs = MagicMock()
        logits = torch.tensor([[0.1, 0.2, 0.7]])  # Predicts the third class
        outputs.logits = logits
        mock_model.return_value = outputs

        # Create a tensor-like object with to() method for tokenizer output
        tokenizer_output = MagicMock()
        tokenizer_output.to = MagicMock(return_value=tokenizer_output)
        mock_tokenizer.return_value = tokenizer_output

        # Create the predictor with mocked HF model ID
        predictor = EmotionPredictor(model_id="test/emotion-model")

        # Replace parts with mocks
        predictor.model = mock_model
        predictor.tokenizer = mock_tokenizer
        predictor.id2label = {0: "happy", 1: "sad", 2: "angry"}
        predictor.label2id = {"happy": 0, "sad": 1, "angry": 2}
        predictor.device = "cpu"  # Ensure we use CPU for tests

        return predictor

def test_is_model_loaded(mock_predictor):
    """Test that the model loaded check works"""
    assert mock_predictor.is_model_loaded() is True

    # Test when model is not loaded
    mock_predictor.model = None
    assert mock_predictor.is_model_loaded() is False

def test_get_labels(mock_predictor):
    """Test retrieving the label list"""
    labels = mock_predictor.get_labels()
    assert set(labels) == {"happy", "sad", "angry"}

    # Test exception when model not loaded
    mock_predictor.model = None  # This will make is_model_loaded() return False
    with pytest.raises(ValueError, match="Model is not loaded"):
        mock_predictor.get_labels()


def test_predict(mock_predictor):
    """Test the prediction functionality"""
    # Configure the model to return specific logits
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 0.2, 0.7]])  # Highest probability for "angry"
    mock_predictor.model.return_value = mock_outputs

    # Reset the necessary attributes
    mock_predictor.id2label = {0: "happy", 1: "sad", 2: "angry"}
    mock_predictor.label2id = {"happy": 0, "sad": 1, "angry": 2}

    result = mock_predictor.predict("I am very upset about this situation.")

    # Verify the prediction result
    assert result["emotion"] == "angry"
    assert result["confidence"] > 0.6  # Should be around 0.7 after softmax
    assert set(result["all_emotions"].keys()) == {"happy", "sad", "angry"}