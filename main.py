import recapper
from video_processing import VideoProcessing
from preproc_files import TransformerToJson
import argparse
import logging
import os
import sys
import re

def parse_recap(entries):
    if not entries:
        return []
    
    # Если передана одна строка, преобразуем в список
    if isinstance(entries, str):
        entries = [entries]
    
    pattern = r'^(.*?)\.json \[(.*?)\] \- (.*?)(?:;|$)'
    results = []
    
    for entry in entries:
        if not isinstance(entry, str):
            continue
            
        match = re.match(pattern, entry)
        if not match:
            continue
            
        filename = match.group(1)
        timecodes = match.group(2).split('–')
        if len(timecodes) != 2:
            continue
            
        start_time, end_time = timecodes[0].strip(), timecodes[1].strip()
        text = match.group(3).strip()
        
        results.append({
            'filename': filename,
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return results

def validate_file(path):
    if not path:
        raise argparse.ArgumentTypeError("Путь не может быть пустым")
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Категории не существует: {path}")

    return path

def read_directory(directory):
    files = os.listdir(directory)

    if not files:
        logging.error(f"Файл не найден: {directory}")
        sys.exit(1)
    
    files_only = [f for f in files if os.path.isfile(os.path.join(directory, f))]

    return files_only

def match_list(list_json, list_mp4):
    cleaned_json = [f.replace('.json', '') for f in list_json]
    cleaned_mp4 = [f.replace('.mp4', '') for f in list_mp4]

    if cleaned_json == cleaned_mp4:
        logging.info("Кол-во и название файлов совпадает")
    else:
        logging.error(f"Кол-во или название файлов не совпадают")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Строит 5-пунктовый рекап эпизода из субтитров.")
    parser.add_argument("--srt", required=True, type = validate_file, help="Путь к srt с субтитрами")
    parser.add_argument("--mp4", required=True, type = validate_file, help="Путь к видео")
    parser.add_argument("--block_duration", type=int, default=30, help="Длительность блока в секундах (по умолчанию 30)")
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as e:
        print(f"Ошибка: {e}")


    files_mp4 = read_directory(args.mp4)
    TransformerToJson(args.srt).process_folder()
    files_json = read_directory(f"{args.srt}\processed_json")
    files_mp4 = sorted(files_mp4) 
    files_json = sorted(files_json) 
    match_list(files_json, files_mp4)

    
    data = recapper.worker(args, files_json)    

    parsed_data = parse_recap(data)

    videos = []
    for i in parsed_data:
        videos.append(VideoProcessing.cutVideo(f"Видео\\{i['filename']}.mp4", i['start'], i['end']))

    video_recap = VideoProcessing.concatenateVideo(videos)
    VideoProcessing.saveVideo('video_recap.mp4', video_recap)
    recapper.write_lines_to_file(parsed_data)


if __name__ == "__main__":
    main()