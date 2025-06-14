# Recap NLP Pipeline

Автоматизация построения кратких текстовых рекапов (summary/recap) эпизодов сериалов по видео и субтитрам.  
Результат пригоден для использования при создании видео-рекапов, промо и удобен для зрителей.

---

## Структура пайплайна

1. **Преобразование субтитров из .srt в .json**
2. **Генерация структурированного recap (5 ключевых моментов) по каждому json с помощью LLM**

---

## Используемые скрипты

### 1. `preproc_files.py`

- Ищет все `.srt`-файлы в указанной папке (`subtitles/`) и конвертирует их в json-файлы с нужной структурой.
- Очищает текст субтитров от неинформативных вставок (например, `(смех)`).
- Результирующие json сохраняются в `subtitles/processed_json/`.

**Запуск:**
```bash
python scripts/srt_to_json.py subtitles
```

### 2. `recaper.py`

- Принимает на вход json-файл (output предыдущего шага).

- Делит субтитры на смысловые блоки (по 30 реплик, параметр можно менять).

- Для каждого блока генерирует краткое summary (с помощью LLM).

- Финальный recap строится по всем summary-блокам — ровно 5 ключевых моментов с тайм-кодами.

- Recap можно использовать для нарезки видеорекапа и интеграции с видео.

**Запуск:**
```bash
python scripts/recapper.py --json subtitles/processed_json/14194_1_1.json
```
Параметр --block-size (по умолчанию 30) меняет “глубину” анализа: меньше — более подробная нарезка.