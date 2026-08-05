FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# PIP_INDEX_URL lets local builds use a mirror; platform builds use the default.
ARG PIP_INDEX_URL=
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src

# Ensure runtime dirs exist (data/ for SQLite, logs/ for app logs)
RUN mkdir -p /app/src/data /app/src/logs

EXPOSE 8000

# main.py lives at src/main.py and exposes `app = create_app()`.
# With PYTHONPATH=/app/src, uvicorn resolves `main:app` correctly.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
