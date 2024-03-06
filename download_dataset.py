```python
# download_dataset.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set your Kaggle API key
api = KaggleApi()
api.authenticate(api_key='your_kaggle_api_key')

# Specify the dataset you want to download
dataset_name = 'mlg-ulb/creditcardfraud'

# Specify the local path where you want to save the dataset
local_path = './credit_card.csv'

# Create the local directory if it doesn't exist
if not os.path.exists(local_path):
    os.makedirs(local_path)

# Download the dataset
api.dataset_download_files(dataset_name, path=local_path, unzip=True)

print(f"The dataset has been downloaded and saved to {local_path}")
