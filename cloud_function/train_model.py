# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage

def train_model(data_file, model_file):
    data = pd.read_csv(data_file)
    X = data.drop('species', axis=1)
    y = data['species']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}")

def upload_model(bucket_name, source_file_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded model to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    BUCKET_NAME = "bucket-demo-project"
    MODEL_FILE_NAME = "model.pkl"
    MODEL_BLOB_NAME = "models/model.pkl"
    DATA_FILE_NAME = "iris_processed.csv"

    train_model(DATA_FILE_NAME, MODEL_FILE_NAME)
    upload_model(BUCKET_NAME, MODEL_FILE_NAME, MODEL_BLOB_NAME)
