FROM python:3.13.2

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

# Set environment variables
# MODEL_ID should be passed during build or run time
ENV MODEL_ID="AfroLogicInsect/emotionClassifier"

# Add caching dir for Hugging Face
RUN mkdir -p /root/.cache/huggingface/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]