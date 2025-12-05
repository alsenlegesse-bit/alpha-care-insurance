import gdown
import os

print("Downloading insurance dataset...")

# Google Drive file ID from the document
file_id = "1QMfobFUaoGab4jvyFzNZ5-u5KaFsve92"
url = f"https://drive.google.com/uc?id={file_id}"

# Create data directory
os.makedirs('data/raw', exist_ok=True)

# Download
output = 'data/raw/insurance_data.csv'
gdown.download(url, output, quiet=False)

print(f"Downloaded to {output}")
