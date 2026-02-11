FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY fishery_model.tflite .
COPY model_meta.json .
COPY pc_inference.py .

# Default: interactive mode
ENTRYPOINT ["python", "pc_inference.py"]
