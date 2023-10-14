import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os 

def parse_srt(filename):
    subtitles = []
    with open(filename, 'r') as srt_file:
        data = srt_file.read()
        subtitle_blocks = data.split('\n\n')
        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                times = lines[1].split(' --> ')
                start_time = times[0]
                start_time_parts = start_time.split(":")
                start_time = (
                    float(start_time_parts[0]) * 3600 + 
                    float(start_time_parts[1]) * 60 +   
                    float(start_time_parts[2].replace(",", ".")) 
                )

                end_time = times[1]
                end_time_parts = end_time.split(":")
                end_time = (
                    float(end_time_parts[0]) * 3600 + 
                    float(end_time_parts[1]) * 60 +   
                    float(end_time_parts[2].replace(",", ".")) 
                )
                text = ' '.join(lines[2:])
                subtitles.append({"start_time": start_time, "end_time": end_time, "text": text})
    return subtitles

# HYPERPARAMS TO MODIFY
season_folder = "../Dataset/S09/"
out = "S09"
# END HYPERPARAMS TO MODIFY

audio_folder = season_folder+"Audio"
subtitle_folder = season_folder+"Subtitles"
clip_folder = season_folder+"Clips"



json_dataset = []
sub_files = os.listdir(subtitle_folder)
for i, file_name in enumerate(os.listdir(audio_folder)):
    if file_name.endswith('.mp3'):
        id_episode = file_name.split(".")[0]

        # Parse subtitles
        path_to_sub = os.path.join(subtitle_folder,sub_files[i])
        subtitles = parse_srt(path_to_sub)

        path_to_clips = clip_folder+"/"+id_episode[0:3]+"/"+id_episode[3:]+"/"
        if not os.path.exists(path_to_clips):
            os.makedirs(path_to_clips)

        data = []
        for i, s in enumerate(subtitles):
            id = id_episode+"_"+"{:06d}".format(i)
            data.append({
                "id": id,
                "duration": s["end_time"] - s["start_time"],
                "text":s["text"],
                "path": "Clips/"+path_to_clips.split("Clips/")[1]+id+".mp3"
            })

            ffmpeg_extract_subclip(os.path.join(audio_folder,file_name), s["start_time"], s["end_time"], targetname=path_to_clips+id+".mp3")
        json_dataset.extend(data)

with open(os.path.join(season_folder,"dataset_"+out+".json"), 'w', encoding='utf-8') as json_file:
    json.dump(json_dataset, json_file, indent=4, ensure_ascii=False)