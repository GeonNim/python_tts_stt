FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Python 경로 설정 후 Gunicorn 실행
CMD ["sh", "-c", "PYTHONPATH=/app gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app"]


