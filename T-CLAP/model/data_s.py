import os
import requests
from datasets import load_dataset

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename

dataset = load_dataset("MLCommons/peoples_speech", split='train', streaming=True)
dataset = dataset.take(10)
output_directory = "/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/data_f/"
dataset.save_to_disk(output_directory)
exit()

# Define the directory where you want to save the files
output_directory = "/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/data_f/"
os.makedirs(output_directory, exist_ok=True)

for example in dataset:
    # This assumes 'url' is a field in your dataset's structure that provides a direct link to the audio file.
    # The actual field name might be different in the dataset you're using.
    audio_url = example['audio']['url']  # Replace 'url' with the actual key, if different.
    
    # Construct a local filename within your desired output directory
    filename = os.path.join(output_directory, os.path.basename(example['audio']['path']))
    
    # Download and save the file
    try:
        download_file(audio_url, filename)
        print(f"File '{filename}' downloaded successfully.")
    except Exception as e:
        print(f"Failed to download the file. Reason: {e}")
