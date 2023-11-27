import json

# Path to the input file where the JSON data is stored
input_file_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAp-main/msclap/data_f/S09E01_updated.json'
# Path to the output file where the updated JSON data will be saved
output_file_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAp-main/msclap/data_f/updated_data_beta.json'

# Read the JSON data from the input file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Identify all unique characters and assign them an index
characters = sorted(set(entry['character'] for entry in data))
character_indices = {character: idx for idx, character in enumerate(characters, start=1)}

# Add the 'character_index' to each entry
for entry in data:
    entry['character_index'] = character_indices[entry['character']]

# Write the updated JSON data to the output file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Updated JSON has been written to {output_file_path}")
