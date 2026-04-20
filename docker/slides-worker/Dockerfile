# Sandboxed PPTX processing worker for meeting-scribe slide translation.
#
# Runs validation, text extraction (python-pptx), and rendering (LibreOffice
# headless) in an isolated container with no network access.
#
# Build:
#   docker build -t meeting-scribe-slides-worker docker/slides-worker/
#
# Runtime (launched by slides/worker.py):
#   docker run --rm --network=none --memory=2g --cpus=2 --read-only \
#     --tmpfs=/tmp:size=500m --user=nobody \
#     -v /path/to/input.pptx:/input/source.pptx:ro \
#     -v /path/to/output:/output:rw \
#     meeting-scribe-slides-worker \
#     python -m meeting_scribe.slides.convert_main \
#       --input=/input/source.pptx --output=/output

FROM python:3.12-slim

# LibreOffice headless for PPTX -> PNG rendering
# Install Microsoft-compatible fonts to prevent layout reflow:
#   Carlito = Calibri substitute (metric-compatible)
#   Caladea = Cambria substitute (metric-compatible)
#   Liberation = Arial/Times/Courier substitutes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-impress \
    libreoffice-common \
    fonts-liberation \
    fonts-liberation2 \
    fonts-crosextra-carlito \
    fonts-crosextra-caladea \
    fonts-freefont-ttf \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    poppler-utils \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# Python deps — only what convert.py needs
RUN pip install --no-cache-dir \
    python-pptx>=1.0.0 \
    Pillow>=10.0.0 \
    lxml>=5.0.0

# Copy only the slides package (convert.py, convert_main.py, models.py)
COPY src/meeting_scribe/slides/ /app/meeting_scribe/slides/
COPY docker/slides-worker/init_package.py /app/meeting_scribe/__init__.py

ENV PYTHONPATH=/app
WORKDIR /app

# Default entry point — overridden by the docker run command from worker.py
CMD ["python", "-m", "meeting_scribe.slides.convert_main", "--help"]
