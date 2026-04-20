# NeMo Sortformer diarization container
#
# Wraps NVIDIA NeMo's Sortformer model in a simple FastAPI server
# that accepts audio chunks and returns speaker segments.
#
# Build:
#   docker build -t nemo-sortformer:latest containers/sortformer/
#
# Run:
#   docker run -d --gpus all --network host --name scribe-diarization \
#     -v /data/huggingface:/root/.cache/huggingface \
#     nemo-sortformer:latest

FROM nvcr.io/nvidia/nemo:24.12

WORKDIR /app

# Install minimal HTTP server deps
RUN pip install --no-cache-dir fastapi uvicorn

COPY server.py /app/server.py

ENV SORTFORMER_PORT=8001
ENV SORTFORMER_MAX_SPEAKERS=4

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
