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
        return None
    full_path = os.path.join('/tmp', destination_file_name)
    blob.download_to_filename(full_path)
    print(f"Downloaded {source_blob_name} to {full_path}")
    return full_path

def evaluate_model(model_file, data_file):
    """Evaluate model performance"""
    model = joblib.load(model_file)
    data = pd.read_csv(data_file)
    X = data.drop('species', axis=1)
    y = data['species']
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Current model accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    MODEL_BLOB_NAME = "models/model.pkl"
    MODEL_FILE_NAME = "model.pkl"
    PROCESSED_DATA_BLOB = "processed_data/iris_processed.csv"
    DATA_FILE_NAME = "iris_processed.csv"
    THRESHOLD = 0.999

    print("Starting model evaluation...")
    
    # Download processed data
    data_file = download_from_gcs(BUCKET_NAME, PROCESSED_DATA_BLOB, DATA_FILE_NAME)
    if not data_file:
        print("Processed data not found. Initial training required.")
        sys.exit(0)

    # Try to download existing model
    model_file = download_from_gcs(BUCKET_NAME, MODEL_BLOB_NAME, MODEL_FILE_NAME)
    if not model_file:
        print("No existing model found. Initial training required.")
        sys.exit(0)

    # Evaluate model
    print(f"Evaluating model against threshold: {THRESHOLD}")
    accuracy = evaluate_model(model_file, data_file)
    
    # Clean up local files
    os.remove(model_file)
    os.remove(data_file)
    
    if accuracy < THRESHOLD:
        print(f"Accuracy ({accuracy:.4f}) below threshold ({THRESHOLD}). Retraining required.")
        sys.exit(1)
    else:
        print(f"Model meets the accuracy threshold. Current: {accuracy:.4f}, Required: {THRESHOLD}")
        sys.exit(0)