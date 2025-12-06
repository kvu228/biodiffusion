import getpass
import os
from zipfile import ZipFile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv


def setup_kaggle_api():
    """Setup and authenticate Kaggle API."""
    load_dotenv()
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    
    if not kaggle_username:
        kaggle_username = getpass.getpass('Enter your Kaggle username: ')
        os.environ['KAGGLE_USERNAME'] = kaggle_username
    if not kaggle_key:
        kaggle_key = getpass.getpass('Enter your Kaggle API key: ')
        os.environ['KAGGLE_KEY'] = kaggle_key
    
    api = KaggleApi()
    api.authenticate()
    return api


def download_mitbih_dataset(api=None, download_path="./datasets/"):
    """Download MITBIH Arrhythmia Database dataset."""
    if api is None:
        api = setup_kaggle_api()
    
    print("=" * 60)
    print("Downloading MITBIH Arrhythmia Database dataset...")
    print("=" * 60)
    
    dataset_name = "shayanfazeli/heartbeat"
    zip_file_path = os.path.join(download_path, "heartbeat.zip")
    extracted_folder_path = os.path.join(download_path, "heartbeat")
    
    # Create directories
    Path(download_path).mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)
    
    # Extract downloaded zip file
    if os.path.exists(zip_file_path):
        print(f"Extracting {zip_file_path} to {extracted_folder_path}...")
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        print("MITBIH dataset downloaded and extracted successfully.")
    else:
        print(f"Warning: {zip_file_path} not found after download.")


def download_unimib_dataset(api=None, download_path="./datasets/"):
    """Download UNIMIB SHAR dataset."""
    if api is None:
        api = setup_kaggle_api()
    
    print("=" * 60)
    print("Downloading UNIMIB SHAR dataset...")
    print("=" * 60)
    
    dataset_name = "wangboluo/unimib-shar-dataset"
    zip_file_path = os.path.join(download_path, "unimib-shar-dataset.zip")
    extracted_folder_path = os.path.join(download_path, "unimib")
    
    # Create directories
    Path(download_path).mkdir(parents=True, exist_ok=True)
    Path(extracted_folder_path).mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)
    
    # Extract downloaded zip file
    if os.path.exists(zip_file_path):
        print(f"Extracting {zip_file_path} to {extracted_folder_path}...")
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        print("UNIMIB dataset downloaded and extracted successfully.")
    else:
        # Try alternative zip file names
        alt_names = ["unimib.zip", "unimib-shar.zip"]
        extracted = False
        for alt_name in alt_names:
            alt_path = os.path.join(download_path, alt_name)
            if os.path.exists(alt_path):
                print(f"Found alternative file: {alt_name}")
                print(f"Extracting {alt_path} to {extracted_folder_path}...")
                with ZipFile(alt_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder_path)
                print("UNIMIB dataset downloaded and extracted successfully.")
                extracted = True
                break
        
        if not extracted:
            print(f"Warning: Zip file not found. Please check downloaded files in {download_path}")
            print("You may need to manually extract the dataset.")


def main():
    """Main function to download datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download biomedical signal datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mitbih', 'unimib', 'all'],
        default='all',
        help='Dataset to download: mitbih, unimib, or all (default: all)'
    )
    
    args = parser.parse_args()
    
    # Setup Kaggle API once
    api = setup_kaggle_api()
    
    # Download selected dataset(s)
    if args.dataset == 'mitbih' or args.dataset == 'all':
        download_mitbih_dataset(api)
    
    if args.dataset == 'unimib' or args.dataset == 'all':
        download_unimib_dataset(api)
    
    print("\n" + "=" * 60)
    print("Download process completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
