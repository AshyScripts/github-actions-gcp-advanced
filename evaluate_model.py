# evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage
import sys

def download_model(bucket_name, model_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_blob_name)
    if not blob.exists():
        print("Model not found in GCS. Retraining is required.")
        return False
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded model to {destination_file_name}")
    return True

def evaluate_model(model_file, data_file):
    model = joblib.load(model_file)
    data = pd.read_csv(data_file)
    X = data.drop('species', axis=1)
    y = data['species']
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    MODEL_BLOB_NAME = "models/model.pkl"
    MODEL_FILE_NAME = "model.pkl"
    DATA_FILE_NAME = "iris_processed.csv"
    THRESHOLD = 0.99

    model_exists = download_model(BUCKET_NAME, MODEL_BLOB_NAME, MODEL_FILE_NAME)
    if not model_exists:
        sys.exit(1)  # Exit code 1 indicates retraining is required

    accuracy = evaluate_model(MODEL_FILE_NAME, DATA_FILE_NAME)
    if accuracy < THRESHOLD:
        print("Accuracy below threshold. Retraining is required.")
        sys.exit(1)
    else:
        print("Model meets the accuracy threshold.")
        sys.exit(0)
