import openai
from tqdm import tqdm

# Initialize the OpenAI API with your key
openai.api_key = "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"

import openai
import json

# Load the JSON data
with open('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAp-main/msclap/data_f/S09E01.json', 'r') as file:
    data = json.load(file)

# Function to get sentiment
def get_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Given the sentiment categories Positive, Negative, Neutral, Angry, Joyful, Surprised, Fearful, Disgusted, Sad, and Trustful, classify the following line into one of them and return only the sentiment as the asnwer:\n\n{text}"}
        ]
    )

    sentiment = response.choices[0].message['content'].strip()
    return sentiment

# Process the data and store sentiment
for idx, entry in enumerate(tqdm(data)):
    sentiment = get_sentiment(entry['line'])
    entry['sentiment'] = sentiment
    entry['index'] = idx

# Optionally, save the updated data to a new JSON file
with open('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAp-main/msclap/data_f/updated_data_alpha.json', 'w') as file:
    json.dump(data, file, indent=4)


