from google.cloud import storage
from pycaret.regression import *
import pandas as pd
import os

def download_blob(project_name: str, 
                    bucket_name: str, 
                    source_blob_name: str, 
                    destination_file_name: str) -> object:
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project_name)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    if destination_file_name is not None: 
      blob.download_to_filename(destination_file_name)

      print(
          "Blob {} downloaded to {}.".format(
              source_blob_name, destination_file_name
          )
      )
    
  
    return blob

def create_dataframe(file_name: str, sep: str=',', var: list=[]) -> pd.DataFrame:
    df = pd.read_csv('./Copy of yield.csv', sep=sep)[var]

    return df

def download_model(model_name: str, 
                    gcp_model_name:str, 
                    CLOUD_PROJECT:str, 
                    bucket_name:str) -> object:
    outfile_name = os.path.join(os.getcwd(), gcp_model_name)
    model_gcp_src = str(model_name)+'.pkl'

    model_downloaded = download_blob(CLOUD_PROJECT, bucket_name, model_gcp_src, outfile_name + '.pkl')
    return model_downloaded, outfile_name

if __name__ == "__main__":
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '' #string kosong ganti jadi path key yang gw kasih

    gcp_model_name = 'rf-gcp-downloaded'
    model_name = 'rf_gcp'
    CLOUD_PROJECT = 'tugas-akhir-332404'
    bucket_name = 'tugas-akhir-332404-ml'

    df = create_dataframe('', sep=';', var=['hari', 'yield']) #string kosong ganti jadi path file csv lu
    _, outfile_name = download_model(model_name, 
                        gcp_model_name,
                        CLOUD_PROJECT,
                        bucket_name)
    
    rf_ta = load_model(outfile_name)
    predict_result = predict_model(rf_ta, data=df)
    predict_result.to_csv('', index=False) #string kosong ganti jadi file output yang lu mau