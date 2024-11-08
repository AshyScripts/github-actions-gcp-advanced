import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage
import sys
import os

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download file from GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if not blob.exists():
        return False
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")
    return True

def evaluate_model(model_file, data_file):
    """Evaluate model performance"""
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
    PROCESSED_DATA_BLOB = "processed_data/iris_processed.csv"
    DATA_FILE_NAME = "iris_processed.csv"
    THRESHOLD = 0.99

    # Download processed data
    if not download_from_gcs(BUCKET_NAME, PROCESSED_DATA_BLOB, DATA_FILE_NAME):
        print("Processed data not found. Run data processing first.")
        sys.exit(1)

    # Try to download existing model
    model_exists = download_from_gcs(BUCKET_NAME, MODEL_BLOB_NAME, MODEL_FILE_NAME)
    
    if not model_exists:
        print("No existing model found. Initial training required.")
        sys.exit(1)

    # Evaluate model
    accuracy = evaluate_model(MODEL_FILE_NAME, DATA_FILE_NAME)
    
    # Clean up local files
    os.remove(MODEL_FILE_NAME)
    os.remove(DATA_FILE_NAME)
    
    if accuracy < THRESHOLD:
        print("Accuracy below threshold. Retraining required.")
        sys.exit(1)
    else:
        print("Model meets the accuracy threshold.")
        sys.exit(0)