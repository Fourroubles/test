
import os
import sys
import json
import argparse
import logging
import re
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

BLOCK_SIZE = 30       # Сколько субтитров в одном смысловом блоке
MODEL_ID = "yandex/YandexGPT-5-Lite-8B-instruct"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_subtitles(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    for s in data.get("subtitles", []):
        txt = s.get("clean_text", s.get("text", "")).strip()
        if txt:
            segments.append({"start": s["start"], "end": s["end"], "text": txt})
    return segments


def split_blocks(segments, block_size=BLOCK_SIZE):
    return [segments[i:i+block_size] for i in range(0, len(segments), block_size)]


def block_timecode(block):
    if not block:
        return "00:00:00–00:00:00"
    return f"{block[0]['start']}–{block[-1]['end']}"


def summarize_block(block, tokenizer, model, device):
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
    return f"[{timecode}] {text}"


def generate_final_recap(block_summaries, tokenizer, model, device):
    prompt = (
        "Вот краткие описания частей эпизода:\n"
        + "\n".join(block_summaries)
        + "\n\nВыдели из них 5 самых важных событий эпизода и составь 5-пунктовый рекап (200–400 слов). "
        "Каждый пункт начинай с тайм-кода из блока и пиши кратко, чтобы зритель вспомнил ключевой сюжет."
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


def main():
    parser = argparse.ArgumentParser(description="Строит 5-пунктовый рекап эпизода из субтитров.")
    parser.add_argument("--json", required=True, help="Путь к JSON с субтитрами")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Размер смыслового блока (по умолчанию 30 субтитров)")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        logging.error(f"Файл не найден: {args.json}")
        sys.exit(1)

    segments = load_subtitles(args.json)
    logging.info(f"Загружено сегментов: {len(segments)}")
    blocks = split_blocks(segments, args.block_size)
    logging.info(f"Создано смысловых блоков: {len(blocks)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                             trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype= torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None ,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Summary для каждого смыслового блока
    block_summaries = []
    for i, block in enumerate(tqdm(blocks, desc="Анализ блоков")):
        if not block:
            continue
        summary = summarize_block(block, tokenizer, model, device)
        block_summaries.append(summary)
        logging.info(f"Блок {i+1}: {summary}")

    # Финальный recap
    recap = generate_final_recap(block_summaries, tokenizer, model, device)
    print("\n=== Итоговый рекап ===\n")
    print(recap)


if __name__ == "__main__":
    main()
