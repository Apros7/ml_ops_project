import os
from google.cloud import storage

def download_model(bucket_name: str, blob_name: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)

    return local_path
