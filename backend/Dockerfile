FROM python:3.10-slim

WORKDIR /app

# Install Python packages separately
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    scikit-learn \
    pydantic

# Install torch from CPU-only index
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY sentiment_engine.py .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT