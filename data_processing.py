import pandas as pd
from google.cloud import storage
import os

def download_data(bucket_name, source_blob_name, destination_file_name):
    """Download data from GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    full_path = os.path.join('/tmp', destination_file_name)
    blob.download_to_filename(full_path)
    print(f"Downloaded {source_blob_name} to {full_path}")
    return full_path

def preprocess_data(input_file, output_file):
    """Preprocess the data and save locally"""
    df = pd.read_csv(input_file)
    df = df.sample(frac=1).reset_index(drop=True)
    full_path = os.path.join('/tmp', output_file)
    df.to_csv(full_path, index=False)
    print(f"Preprocessed data saved to {full_path}")
    return full_path

def upload_processed_data(bucket_name, source_file_name, destination_blob_name):
    """Upload processed data to GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded processed data to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    SOURCE_BLOB_NAME = "data/iris.csv"
    DESTINATION_FILE_NAME = "iris.csv"
    PROCESSED_FILE_NAME = "iris_processed.csv"
    PROCESSED_BLOB_NAME = "processed_data/iris_processed.csv"

    # Download raw data
    input_file = download_data(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
    
    # Preprocess data
    processed_file = preprocess_data(input_file, PROCESSED_FILE_NAME)
    
    # Upload processed data
    upload_processed_data(BUCKET_NAME, processed_file, PROCESSED_BLOB_NAME)
    
    # Clean up local files
    os.remove(input_file)
    os.remove(processed_file)