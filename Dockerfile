# Base python image
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data_processing.py .
COPY train_model.py .
COPY evaluate_model.py .

EXPOSE 8080

ENV NAME World

CMD ["python", "train_model.py"]