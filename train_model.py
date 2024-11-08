import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage
import os

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download file from GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def train_model(data_file, model_file):
    """Train the model and save locally"""
    data = pd.read_csv(data_file)
    X = data.drop('species', axis=1)
    y = data['species']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload file to GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    MODEL_FILE_NAME = "model.pkl"
    MODEL_BLOB_NAME = "models/model.pkl"
    PROCESSED_DATA_BLOB = "processed_data/iris_processed.csv"
    DATA_FILE_NAME = "iris_processed.csv"

    # Download processed data
    download_from_gcs(BUCKET_NAME, PROCESSED_DATA_BLOB, DATA_FILE_NAME)
    
    # Train model
    train_model(DATA_FILE_NAME, MODEL_FILE_NAME)
    
    # Upload model to GCS
    upload_to_gcs(BUCKET_NAME, MODEL_FILE_NAME, MODEL_BLOB_NAME)
    
    # Clean up local files
    os.remove(MODEL_FILE_NAME)
    os.remove(DATA_FILE_NAME)