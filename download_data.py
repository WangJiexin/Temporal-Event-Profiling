from google_drive_downloader import GoogleDriveDownloader as gdd
import os

def download_data():
	gdd.download_file_from_google_drive(file_id='1aAKtj8HRktyt6VrQfkzHMf-k_-hE5U_j',
	                                    dest_path='./data/input_data.zip',
	                                    unzip=True)
	os.remove('./data/input_data.zip')
	
if __name__ == "__main__":
    download_data()
