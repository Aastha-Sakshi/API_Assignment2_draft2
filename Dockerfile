FROM python:3.11-slim

# OCR is docTR (pip), so there is no OCR binary to install. These are only
# opencv's shared-library needs on a slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# torch is the heavy layer — install deps before copying source so an edit to
# app/ does not invalidate a ~2GB pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY streamlit_app.py .

# Models download on first use. For a reproducible image, bake them in here
# instead — first-request latency otherwise includes a multi-hundred-MB pull.
ENV HF_HOME=/srv/.cache/huggingface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
