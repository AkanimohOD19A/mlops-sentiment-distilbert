from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import os
from app.ml.predictor import EmotionPredictor

app = FastAPI(title="Emotion Classification API")

# Get model ID from environment variable or use default
model_id = os.environ.get("MODEL_ID", "AfroLogicInsect/emotionClassifier")

# Initialize the predictor with Hugging Face model ID
predictor = EmotionPredictor(model_id=model_id)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Emotion Classification API",
        "model": model_id
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_model_loaded(),
        "model_source": model_id
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict the emotion of the input text"""
    try:
        result = predictor.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/labels", response_model=List[str])
def get_labels():
    """Return the list of available emotion labels"""
    return predictor.get_labels()