import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionPredictor:
    def __init__(self, model_path: Optional[str] = None, model_id: Optional[str] = None):
        """
        Initialize the emotion predictor with a pre-trained model

        Args:
            model_path: Path to the saved model directory. If None, will use the
                       environment variable MODEL_PATH or the default "./model_export"
            model_id: HuggingFace Hub model ID (e.g., "your-username/emotion-classifier")
                     If provided, will load from HuggingFace instead of local path
        """
        self.model_path = model_path or os.environ.get("MODEL_PATH", "./model_export")
        self.model_id = model_id or os.environ.get("MODEL_ID")
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer from local directory or HuggingFace Hub"""
        try:
            # If model_id is provided, load from HuggingFace Hub
            if self.model_id:
                print(f"Loading model from HuggingFace Hub: {self.model_id}")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

                # Get label mappings from model config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
            else:
                # Load from local path
                print(f"Loading model from local path: {self.model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

                # Load label mappings
                mappings_path = os.path.join(self.model_path, 'label_mappings.json')
                if os.path.exists(mappings_path):
                    with open(mappings_path, 'r') as f:
                        mappings = json.load(f)
                        self.label2id = mappings['label2id']
                        self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
                else:
                    # If no mappings file, use the model config
                    self.id2label = self.model.config.id2label
                    self.label2id = self.model.config.label2id

            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None and self.tokenizer is not None

    def get_labels(self) -> List[str]:
        """Get the list of emotion labels"""
        if not self.is_model_loaded():
            raise ValueError("Model is not loaded")
        return list(self.label2id.keys())

    def predict(self, text: str) -> Dict:
        """
        Predict the emotion of the input text

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction results
        """
        if not self.is_model_loaded():
            raise ValueError("Model is not loaded")

        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

        # Get the predicted class and confidence
        predicted_class_id = int(np.argmax(probabilities))
        predicted_emotion = self.id2label[predicted_class_id]
        confidence = float(probabilities[predicted_class_id])

        # Create the response with all emotion probabilities
        all_emotions = {
            self.id2label[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_emotions": all_emotions
        }