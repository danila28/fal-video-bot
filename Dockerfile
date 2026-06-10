FROM python:3.12-slim

WORKDIR /app

# ffmpeg для пост-обработки видео (mux, karaoke, grade, speed).
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/static

CMD ["python", "main.py"]
