FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY openphonic ./openphonic
COPY config ./config

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "openphonic.main:app", "--host", "0.0.0.0", "--port", "8000"]
