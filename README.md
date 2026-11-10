# OCR Service (EasyOCR + Tesseract)

HTTP-сервис распознавания документов. Два движка на выбор: **EasyOCR** (PyTorch, GPU) и **Tesseract** (CPU).

## Возможности
- Распознавание русского и английского текста (`ru`, `en`)
- Поддержка форматов: JPG, PNG, BMP, TIFF, **PDF** (многостраничные)
- Два движка на выбор (параметр `engine`):
  - **EasyOCR** (по умолчанию) — PyTorch, GPU-ускорение (RTX 3070 Ti, CUDA 12.6), выше точность по кириллице
  - **Tesseract** 5.5 (`rus+eng`) — CPU, быстрее, стабильнее по латинице
- **Автоопределение ориентации страницы**: каждая страница (или изображение) пробуются в 4 ориентациях (0/90/180/270°), выбирается та, где выше доля слов с уверенностью ≥ 80%. Обрабатывает:
  - PDF-страницы с атрибутом `/Rotate` (деротация при рендере)
  - Изображения с EXIF-тегом Orientation (фото с телефона)
  - Сканы/страницы, повёрнутые на 90°/270° (ландшафт на портретной странице) или перевёрнутые на 180°

## API

### `GET /health`
```json
{"status":"ok","torch_version":"2.14.1+cu126","cuda_available":true,"gpu":"NVIDIA GeForce RTX 3070 Ti","easyocr_loaded":true,"tesseract_available":true}
```

### `POST /recognize`
Отправить изображение или PDF. Параметры:
- `file` — файл (JPG, PNG, BMP, TIFF, PDF)
- `engine` — `easyocr` (по умолчанию) или `tesseract`

```bash
# EasyOCR (по умолчанию)
curl -X POST http://ocr.example.com:8100/recognize -F "file=@document.pdf"

# Tesseract
curl -X POST http://ocr.example.com:8100/recognize -F "file=@document.pdf" -F "engine=tesseract"
```

Ответ:
```json
{
  "engine": "easyocr",
  "pages": 1,
  "processing_time_s": 0.322,
  "text_count": 5,
  "full_text": "Привет мир ...",
  "results": [
    {"text": "...", "confidence": 0.99, "bbox": [[x,y],...]}
  ],
  "orientations": [0]
}
```

`orientations` — массив (по одному значению на страницу/изображение): угол поворота в градусах (0/90/180/270), который был применён для приведения страницы к нормальному положению. `0` — страница не поворачивалась.

> ⚠️ Автоопределение ориентации увеличивает время обработки: для каждой страницы делается до 4 проходов OCR (ранний выход, если исходная ориентация уже даёт ≥ 85% слов с высокой уверенностью). Для Tesseract это ~2–4× к времени на страницу, для EasyOCR — сопоставимо (GPU).

## Структура
```
server.py         — FastAPI приложение (EasyOCR + Tesseract)
easyocr.service   — systemd unit file
requirements.txt  — зафиксированные версии пакетов
```

## Замечания
- Tesseract работает на CPU и не зависит от GPU; бинарник ставится в систему (apt), а `pytesseract` — в venv.
- При первом обращении к `engine=tesseract` сервис проверяет наличие языковых данных (`rus`, `eng`) и возвращает ошибку 500 с подсказкой, если их не хватает.
- Уверенность Tesseract приводится к формату 0–1 (как у EasyOCR), bbox — в тех же координатах `[x, y]`.
