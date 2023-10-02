from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

audio_file = "audio_ep1.mp3"
subtitles_file = "subtitles.txt"

subtitles = []
lines = []
with open(subtitles_file, "r") as subtitle_file:
    lines = subtitle_file.read().splitlines()

for i in range(len(lines)):
    line = lines[i]
    if "-->" in line:
        try:
            start_time_str, end_time_str = line.strip().split(" --> ")
            print(line)
            
            start_time_parts = start_time_str.split(":")
            end_time_parts = end_time_str.split(":")
            
            start_time = (
                float(start_time_parts[0]) * 3600 + 
                float(start_time_parts[1]) * 60 +   
                float(start_time_parts[2].replace(",", ".")) 
            )
            
            end_time = (
                float(end_time_parts[0]) * 3600 +    
                float(end_time_parts[1]) * 60 +        
                float(end_time_parts[2].replace(",", ".")) 
            )
            
            subtitles.append({"start": start_time, "end": end_time})
        except ValueError:
            print(f"Skipping invalid subtitle format at line {i + 1}: {line}")

output_directory = "output_audio_clips/"

for i, subtitle in enumerate(subtitles):
    start_time = subtitle["start"]
    end_time = subtitle["end"]
    output_file = f"{output_directory}clip_{i + 1}.mp3"
    
    ffmpeg_extract_subclip(audio_file, start_time, end_time, targetname=output_file)

print("Audio clips have been successfully split.")
