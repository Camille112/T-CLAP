import re
import csv
from collections import defaultdict

# Function to calculate Jaccard similarity between two sets of words
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union > 0 else 0.0

# Read the subtitle file
with open('subtitles.txt', 'r') as subtitle_file:
    subtitle_lines = subtitle_file.readlines()

# Read the script file
with open('script.txt', 'r') as script_file:
    script_lines = script_file.readlines()

current_subtitle = ""
current_speaker = ""
subtitle_data = defaultdict(list)  # Use a defaultdict to group subtitles by timestamp

for line in subtitle_lines:
    line = line.strip()
    if re.match(r'^\d+$', line):
        continue
    elif re.match(r'^\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+$', line):
        timestamp = line
        current_subtitle = ""
    elif line:  # Check if it's a non-empty line (subtitle text)
        current_subtitle += line + " "

        # Tokenize subtitle and calculate Jaccard similarity with script lines
        subtitle_words = set(current_subtitle.split())
        matching_lines = []

        for script_line in script_lines:
            script_words = set(script_line.split())
            similarity = jaccard_similarity(subtitle_words, script_words)
            
            if similarity >= 0.3:  # You can adjust the threshold as needed
                current_speaker = script_line.split(": ")[0]
                matching_lines.append(similarity)

        if matching_lines:
            # Keep the line with the highest similarity
            max_similarity = max(matching_lines)
            subtitle_data[timestamp].append({
                "Speaker": current_speaker,
                "Subtitle": line,
                "Similarity": max_similarity
            })

# Process the grouped data to calculate the mean similarity
combined_subtitle_data = []
for timestamp, data_list in subtitle_data.items():
    if data_list:
        total_similarity = sum(item["Similarity"] for item in data_list)
        mean_similarity = total_similarity / len(data_list)
        combined_subtitle_data.append({
            "Timestamp": timestamp,
            "Speaker": data_list[0]["Speaker"],
            "Subtitle": " ".join(item["Subtitle"] for item in data_list),
            "Similarity": mean_similarity
        })

# Export the subtitle data to CSV
with open('output.csv', 'w', newline='') as csv_file:
    fieldnames = ["Timestamp", "Speaker", "Subtitle", "Similarity"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    
    writer.writeheader()
    for data in combined_subtitle_data:
        writer.writerow(data)
