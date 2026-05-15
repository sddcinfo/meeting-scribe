#!/usr/bin/env bash
# Install lingua-language-detector into the project venv (~170 MB wheel).
#
# Used by both:
#   * slides/convert.py::_detect_via_lingua  (slide language detection)
#   * language_correction.py::correct_segment_language  (ASR post-correction)
#
# Wrapped as a script so it stays inside the .venv guard rails.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pip install --upgrade 'lingua-language-detector>=2.2.0'
