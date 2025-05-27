import re
import pandas as pd
import torch
import numpy as np
from glob import glob
import os
import json
import sys
class TransformerToJson:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.out_dir = os.path.join(folder_path, "processed_json")
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def parse_filename(filename):
        basename = os.path.basename(filename)
        m = re.match(r"(\d+)_(\d+)_(\d+)\.srt", basename)
        if not m:
            raise ValueError(f"Неправильное имя файла: {filename}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    @staticmethod
    def parse_srt(filepath):
        with open(filepath, encoding='utf-8-sig') as f:
            content = f.read()
        blocks = re.split(r'\n\s*\n', content)
        subtitles = []
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                index = lines[0].strip()
                try:
                    index = int(index)
                except Exception:
                    index = -1
                times = lines[1].strip()
                text = " ".join(lines[2:]).strip()
                m = re.match(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', times)
                if m:
                    start, end = m.groups()
                else:
                    start, end = None, None
                subtitles.append({
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": text
                })
        return subtitles

    @staticmethod
    def clean_subtitle_text(text):
        return re.sub(r'\([^)]+\)', '', text).strip()

    def process_srt_file(self, filepath):
        series_id, season, episode = self.parse_filename(filepath)
        data = {
            "series_id": series_id,
            "season": season,
            "episode": episode,
            "filename": os.path.basename(filepath),
            "subtitles": []
        }
        for sub in self.parse_srt(filepath):
            clean_text = self.clean_subtitle_text(sub['text'])
            if clean_text:  # Только непустые реплики
                entry = {
                    "index": sub["index"],
                    "start": sub["start"],
                    "end": sub["end"],
                    "text": sub["text"],
                    "clean_text": clean_text
                }
                data["subtitles"].append(entry)
        out_path = os.path.join(self.out_dir, os.path.basename(filepath).replace('.srt', '.json'))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")

    def process_folder(self):
        files = sorted(glob(os.path.join(self.folder_path, "*.srt")))
        for file in files:
            self.process_srt_file(file)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else 'субтитры'
    TransformerToJson(folder).process_folder()
