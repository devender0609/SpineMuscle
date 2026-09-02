FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl dcm2niix && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY frozen /opt/spinemuscle/frozen
COPY pipeline /opt/spinemuscle/frozen/pipeline
ENV SPINEMUSCLE_FROZEN_DIR=/opt/spinemuscle/frozen
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
