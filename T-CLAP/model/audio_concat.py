import wave
from pydub import AudioSegment
import random
import csv
import itertools
import csv
from pydub import AudioSegment
import json
from pydub import AudioSegment

# Path to your JSON file
filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
root_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'
s_path = './TemporalAudio_2/'

def process_audio_operations(file_data):

    for group in file_data:
        
        text_a = group["A"][0]
        path_a = group["A"][1]

        text_b = group["B"][0]
        path_b = group["B"][1]

        text_a_before_b = group["A before B"][0]
        path_a_before_b = group["A before B"][1]

        text_b_before_a = group["B before A"][0]
        path_b_before_a = group["B before A"][1]

        text_a_while_b = group["A while B"][0]
        path_a_while_b = group["A while B"][1]

        # Load audio files
        audio_a = AudioSegment.from_file(root_path+path_a)
        audio_a.export(s_path+path_a, format="wav")

        audio_b = AudioSegment.from_file(root_path+path_b)
        audio_b.export(s_path+path_b, format="wav")

        audio_a_before_b = audio_a + audio_b
        audio_a_before_b.export(path_a_before_b, format="wav")

        audio_b_before_a = audio_b+audio_a
        audio_b_before_a.export(path_b_before_a, format="wav")

        audio_a_while_b = audio_a.overlay(audio_b)
        audio_a_while_b.export(path_a_while_b, format="wav")

# Read JSON data from the file
with open(filepath, 'r') as file:
    file_data = json.load(file)

# Call the function with the JSON data
process_audio_operations(file_data)
