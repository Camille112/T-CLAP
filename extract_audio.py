import os
import subprocess

def convert_video_to_audio(input_video, output_audio):
    command = f"ffmpeg -i {input_video} -ab 640k {output_audio}"
    subprocess.run(command, shell=True)

def convert_videos_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mkv'):
            input_video_path = os.path.join(input_folder, file_name)
            audio_name = file_name.split('.')[1]
            audio_name = audio_name+".mp3"
            
            output_audio_path = os.path.join(output_folder, audio_name)
            convert_video_to_audio(input_video_path, output_audio_path)

# HYPERPARAMS TO MODIFY
input_folder = "../Dataset/S09/Video"
output_folder = "../Dataset/S09/Audio"
# END HYPERPARAMS TO MODIFY

convert_videos_in_folder(input_folder, output_folder)
