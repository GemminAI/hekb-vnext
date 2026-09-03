FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEKB_HOST=0.0.0.0 \
    HEKB_PORT=8300 \
    HEKB_DATA_DIR=/app/experience

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY store_service.py .
COPY vendor ./vendor

RUN mkdir -p /app/experience/objects /app/experience/relations /app/experience/lineages

EXPOSE 8300

CMD ["python", "store_service.py"]
