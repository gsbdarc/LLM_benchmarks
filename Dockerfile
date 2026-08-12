FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 harness \
    && mkdir -p /app/.harness-data \
    && chown -R harness:harness /app
COPY --chown=harness:harness pdf_harness ./pdf_harness
COPY --chown=harness:harness .streamlit ./.streamlit

USER harness

CMD ["sh", "-c", "streamlit run pdf_harness/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
