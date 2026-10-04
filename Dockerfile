FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    APP_ENV=production DATA_DIR=/data HOST=0.0.0.0 PORT=8000

WORKDIR /srv
# INSTALL_ML=1 (default) bundles SBERT + RoBERTa + GPT-2 (CPU torch, ~2 GB image).
# Build with --build-arg INSTALL_ML=0 for the lightweight statistical-only image.
ARG INSTALL_ML=1
COPY requirements.txt requirements-ml.txt ./
RUN if [ "$INSTALL_ML" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-ml.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi

COPY app ./app
COPY sample_docs ./sample_docs
COPY server.py .
COPY scripts ./scripts
COPY evaluation ./evaluation

# Bake model weights into the image so start-up needs no download (skipped when INSTALL_ML=0)
ENV HF_HOME=/opt/hf
RUN if [ "$INSTALL_ML" = "1" ]; then python scripts/download_models.py; fi

# Run as an unprivileged user; /data is where the database lives (mount a volume here)
RUN useradd --create-home appuser && mkdir /data && chown appuser /data && chmod -R a+rX /opt/hf 2>/dev/null || true
USER appuser
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

# One worker on purpose: login/analyze rate limits are kept in memory.
# --proxy-headers lets the app see real client IPs behind your HTTPS proxy.
CMD ["sh", "-c", "uvicorn server:app --host $HOST --port $PORT --proxy-headers --forwarded-allow-ips='*'"]
