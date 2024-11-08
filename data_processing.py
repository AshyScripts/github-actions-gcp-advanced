# data_processing.py

import pandas as pd
from google.cloud import storage

def download_data(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    SOURCE_BLOB_NAME = "data/iris.csv"
    DESTINATION_FILE_NAME = "iris.csv"
    PROCESSED_FILE_NAME = "iris_processed.csv"

    download_data(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
    preprocess_data(DESTINATION_FILE_NAME, PROCESSED_FILE_NAME)
