FROM python:3.11-slim

# Системные зависимости:
#   tesseract-ocr — нужен для .ocr
#   ffmpeg        — нужен pydub/SpeechRecognition для .vm (голосовые/кружки)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# TESSERACT_CMD не задаём — на Linux бинарник уже лежит в PATH
# после apt install (просто "tesseract"), pytesseract найдёт его сам.

CMD ["python", "bot.py"]
