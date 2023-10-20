import wave
from pydub import AudioSegment
import random
import csv
import itertools

import csv
from pydub import AudioSegment

# Path to the directory containing the audio files and the CSV
path_audio_files = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'
path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/meta/'
# New directory to save combined audio files
path_new = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/combo_audio/'

# Function to create a new target value based on your logic
def create_new_target(target1, target2):
    # Add your logic here to create a new target value
    # For the example, we'll just create a string from the two targets
    tar = int(target1)+int(target2)
    return f"{tar}"

# Function to create a new category description
def create_new_category(category1, category2):
    return f"{category1} before {category2}"

# Read the original CSV file
with open(path+'original.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Skip header
    header = next(reader)
    count=0

    # Assuming the header of your CSV files are 'filename', 'target', 'category'
    # as you mentioned in your question.

    # Preparing the new table
    new_rows = []
    for row1, row2 in itertools.combinations(reader, 2):  # Combining every two rows
        count+=1
        #new_filename = f"{row1[0]}_{row2[0]}"
        new_filename = row1[0].replace('.wav', '') + '_' + row2[0]  # combine filenames
        new_target = create_new_target(row1[2], row2[2])
        new_category = create_new_category(row1[3], row2[3])
        
        new_rows.append([new_filename, new_target, new_category])

# Number of random elements you want
num_random_elements = 1000

# Check if the original list has enough elements to sample from
if len(new_rows) >= num_random_elements:
    # Using random.sample to pick 100 random items from 'old_list'
    new_list = random.sample(new_rows, num_random_elements)

# Write the new table to a new CSV file
with open(path
+'combined.csv', 'w', newline='') as new_csv_file:
    writer = csv.writer(new_csv_file)
    # Write the header
    writer.writerow(['filename', 'target', 'category'])
    # Write the content
    writer.writerows(new_list)

print(f"Combined CSV created with {len(new_list)} rows.")

def combine_audio_files(file1, file2, combined_filename):
    # Load the audio files
    audio1 = AudioSegment.from_wav(path_audio_files + file1)
    audio2 = AudioSegment.from_wav(path_audio_files + file2)

    # Concatenate the two audio files
    combined_audio = audio1 + audio2

    # Save the combined audio to a new file
    combined_audio.export(path_new + combined_filename, format='wav')

# Process the CSV file
with open(path+'combined.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row
    next(csv_reader)

    # Process each row in the CSV file
    for row in csv_reader:
        # Split the 'filename' field by the underscore character
        files = row[0].split('_')

        # If there are exactly two filenames, proceed
        if len(files) == 2:
            file1, file2 = files
            file1 = file1+'.wav'
            combined_filename = f"{file1[:-4]}_{file2}"  # Removing '.wav' from file1 for the new filename
            
            try:
                # Combine the audio files and save them with a new filename
                combine_audio_files(file1, file2, combined_filename)
                print(f"Files {file1} and {file2} combined successfully into {combined_filename}")
            except Exception as e:
                print(f"An error occurred while combining files {file1} and {file2}: {e}")
        else:
            print(f"Unexpected filename format in row: {row}")
