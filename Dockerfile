FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN npm install \
    && npm --prefix apps/web install \
    && python -m pip install --upgrade pip \
    && python -m pip install -e .
RUN npm run build

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
