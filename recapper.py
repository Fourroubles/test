import os
import sys
import json
import argparse
import logging
import re
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "yandex/YandexGPT-5-Lite-8B-instruct"
BLOCK_DURATION_SEC = 30  # Длительность блока в секундах

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def time_to_seconds(time_str):
    """Конвертирует строку времени в секунды с миллисекундами."""
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_time(total_seconds):
    """Конвертирует секунды обратно в строку времени."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def load_subtitles(json_path):
    """Загружает субтитры из JSON файла."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    for s in data.get("subtitles", []):
        txt = s.get("clean_text", s.get("text", "")).strip()
        if txt:
            segments.append({"start": s["start"], "end": s["end"], "text": txt})
    return segments


def split_blocks(segments, block_duration_sec=BLOCK_DURATION_SEC):
    """Разделяет субтитры на блоки по временным интервалам."""
    if not segments:
        return []
    
    blocks = []
    current_block = []
    
    # Время начала первого блока
    first_start = time_to_seconds(segments[0]['start'])
    current_block_end = first_start + block_duration_sec
    
    for segment in segments:
        segment_start = time_to_seconds(segment['start'])
        segment_end = time_to_seconds(segment['end'])
        
        # Если сегмент полностью в текущем блоке
        if segment_end <= current_block_end:
            current_block.append(segment)
        else:
            # Если сегмент пересекает границу блока
            if segment_start < current_block_end:
                # Добавляем часть сегмента в текущий блок
                split_segment = segment.copy()
                split_segment['end'] = seconds_to_time(current_block_end)
                current_block.append(split_segment)
                
                # Начинаем новый блок с оставшейся части
                blocks.append(current_block)
                current_block = [segment.copy()]
                current_block[0]['start'] = seconds_to_time(current_block_end)
                current_block_end += block_duration_sec
            else:
                # Просто начинаем новый блок
                blocks.append(current_block)
                current_block = [segment]
                current_block_end = segment_start + block_duration_sec
    
    if current_block:
        blocks.append(current_block)
    
    return blocks


def block_timecode(block):
    """Генерирует таймкод для блока."""
    if not block:
        return "00:00:00–00:00:00"
    return f"{block[0]['start']}–{block[-1]['end']}"


def summarize_block(block, tokenizer, model, device, file_name):
    """Создает краткое описание для блока субтитров."""
    timecode = block_timecode(block)
    lines = [f"[{x['start']}–{x['end']}] {x['text']}" for x in block]
    prompt = (
        f"Вот часть субтитров эпизода с тайм-кодами:\n"
        + "\n".join(lines)
        + f"\n\nОпиши в 1-2 предложениях самое главное событие или конфликт этого фрагмента ({timecode}). "
        "Не выдумывай ничего, используй только информацию из этих субтитров. Если ничего важного не происходит — опиши атмосферу или общий характер диалога."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                      max_length=tokenizer.model_max_length).to(device)
    gen_cfg = GenerationConfig(max_new_tokens=100, do_sample=True,
                              temperature=0.7, top_p=0.9,
                              repetition_penalty=1.1)
    out = model.generate(**inputs, generation_config=gen_cfg,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:],
                           skip_special_tokens=True).strip()
    return f"{file_name}[{timecode}] {text}"


def generate_final_recap(block_summaries, tokenizer, model, device):
    """Генерирует финальный рекап на основе всех блоков."""
    prompt = (
        "Вот краткие описания частей эпизода:\n"
        + "\n".join(block_summaries)
        + "\n\nВыдели из них 5 самых важных событий эпизода и составь 5-пунктовый рекап (200–400 слов). "
        "Каждый пункт начинай с тайм-кода и соотвесвующего фрагменту файлу из блока и пиши кратко, чтобы зритель вспомнил ключевой сюжет."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                      max_length=tokenizer.model_max_length).to(device)
    gen_cfg = GenerationConfig(max_new_tokens=600, do_sample=True,
                              temperature=0.7, top_p=0.9,
                              repetition_penalty=1.1)
    out = model.generate(**inputs, generation_config=gen_cfg,
                         pad_token_id=tokenizer.eos_token_id)
    recap = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:],
                           skip_special_tokens=True).strip()
    return recap


def worker(args, files):
    """Основная функция обработки."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                            trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Summary для каждого временного блока
    block_summaries = []
    for file in files:
        segments = load_subtitles(f"{args.json}/processed_json/{file}")
        logging.info(f"Загружен файл: {file}")
        logging.info(f"Загружено сегментов: {len(segments)}")
        blocks = split_blocks(segments, args.block_duration)
        logging.info(f"Создано временных блоков: {len(blocks)}")
        for i, block in enumerate(tqdm(blocks, desc="Анализ блоков")):
            if not block:
                continue
            summary = summarize_block(block, tokenizer, model, device, file)
            block_summaries.append(summary)
            logging.info(f"Блок {i+1} ({block_timecode(block)}): {summary}")

    # Финальный recap
    recap = generate_final_recap(block_summaries, tokenizer, model, device)
    print("Финальный рекап")
    print(recap)
    return recap

def write_lines_to_file(lines):
    """
    Записывает список строк в файл, каждая строка с новой строки.

    :param lines: список строк для записи
    :param filename: имя файла для сохранения
    """
    with open('text_recap.txt', 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line['text'] + '\n')


