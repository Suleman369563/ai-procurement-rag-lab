FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV wheels may require runtime libs even in headless mode.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
