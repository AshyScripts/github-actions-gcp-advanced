import pandas as pd
from google.cloud import storage
import os

def download_data(bucket_name, source_blob_name, destination_file_name):
    """Download data from GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def preprocess_data(input_file, output_file):
    """Preprocess the data and save locally"""
    df = pd.read_csv(input_file)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

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
    download_data(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
    
    # Preprocess data
    preprocess_data(DESTINATION_FILE_NAME, PROCESSED_FILE_NAME)
    
    # Upload processed data
    upload_processed_data(BUCKET_NAME, PROCESSED_FILE_NAME, PROCESSED_BLOB_NAME)
    
    # Clean up local files
    os.remove(DESTINATION_FILE_NAME)
    os.remove(PROCESSED_FILE_NAME)