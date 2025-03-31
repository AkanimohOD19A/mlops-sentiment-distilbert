# Emotion Classification API

This project implements a machine learning service that classifies text into emotions using a fine-tuned DistilBERT model.

## Architecture Overview

![Architecture Diagram](path/to/your/diagram.png)

The system consists of:
1. A fine-tuned DistilBERT model hosted on Hugging Face Hub
2. A FastAPI service for model serving
3. CI/CD pipeline using GitHub Actions
4. Demo deployment on Render

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (for local container testing)
- Hugging Face account (if uploading your own model)

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-classifier.git
cd emotion-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API locally:
```bash
# Using the Hugging Face model
MODEL_ID="your-username/emotion-classifier" uvicorn app.main:app --reload

# OR using a local model
MODEL_PATH="./model_export" uvicorn app.main:app --reload
```

4. Visit the API documentation at http://localhost:8000/docs

### Running Tests

```bash
pytest --cov=app tests/
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

1. **Testing**: Automatically runs tests on every push and pull request
2. **Docker Build**: Builds a Docker image for the service
3. **Deployment**: Deploys the service to a demo environment on Render

## Model Training & Experiment Tracking

The emotion classification model was trained on [dataset description]. The training process used MLflow for experiment tracking.

For details on the training process, see the [training notebook](notebooks/training_notebook.ipynb).

## Deployment

The CI/CD pipeline automatically deploys the application to Render:
- Demo URL: https://your-app-name.onrender.com

## Project Structure

```
emotion-classifier/
├── .github/workflows/      # GitHub Actions workflows
├── app/                    # FastAPI application
│   ├── main.py             # Main API endpoints
│   └── ml/                 # ML code
│       └── predictor.py    # Model prediction class
├── tests/                  # Unit tests
├── Dockerfile              # Docker configuration
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

### NOTES
-> sOFTMAX Calc. `[0.1, 0.2, 0.7]`
The softmax calculation takes exponentials of each value and then normalizes them to sum to 1. When the differences between values are small, the probabilities are more evenly distributed.