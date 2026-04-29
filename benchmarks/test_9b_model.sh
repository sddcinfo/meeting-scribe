#!/bin/bash
# Benchmark Qwen3.5-9B-FP8 translation quality vs 35B baseline.
#
# This script:
# 1. Stops the 35B translation container
# 2. Starts a 9B translation container on port 8000
# 3. Runs quality and throughput benchmarks
# 4. Restarts the 35B container
#
# Results saved to benchmarks/results/
#
# Usage: bash benchmarks/test_9b_model.sh

set -e

MODEL_9B="lovedheart/Qwen3.5-9B-FP8"
VLLM_IMAGE="${SCRIBE_VLLM_IMAGE:-vllm/vllm-openai:latest}"

echo "=== BENCHMARKING Qwen3.5-9B-FP8 vs 35B ==="
echo "This will temporarily stop the translation container."
echo ""

# 1. Stop 35B
echo ">>> Stopping 35B translation container..."
docker stop scribe-translation 2>/dev/null || true
docker rm scribe-translation 2>/dev/null || true
sleep 2

# 2. Start 9B on port 8000
echo ">>> Starting 9B translation container..."
docker run -d \
  --name scribe-translation-9b \
  --gpus all \
  --network host \
  --shm-size 16g \
  -v /data/huggingface:/root/.cache/huggingface \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  "$VLLM_IMAGE" \
  /bin/bash -c "vllm serve $MODEL_9B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --max-num-seqs 32 \
    --enforce-eager \
    --load-format safetensors"

# 3. Wait for health
echo ">>> Waiting for 9B model to load..."
for i in $(seq 1 60); do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8010/health 2>/dev/null)
  if [ "$status" = "200" ]; then
    echo ">>> HEALTHY after $((i*10))s"
    break
  fi
  sleep 10
done

# 4. Run quality benchmarks
echo ""
echo ">>> Running JA→EN quality benchmark..."
.venv/bin/python benchmarks/translation_benchmark.py \
  --url http://localhost:8010 \
  --model-name "Qwen3.5-9B-FP8" \
  --direction ja_to_en

echo ""
echo ">>> Running EN→JA quality benchmark..."
.venv/bin/python benchmarks/translation_benchmark.py \
  --url http://localhost:8010 \
  --model-name "Qwen3.5-9B-FP8" \
  --direction en_to_ja

echo ""
echo ">>> Running concurrency benchmark..."
.venv/bin/python benchmarks/vllm_optimization.py \
  --url http://localhost:8010 \
  --label "Qwen3.5-9B-FP8" \
  --concurrency "1,4,8,16"

# 5. Compare results
echo ""
echo "=== QUALITY COMPARISON ==="
.venv/bin/python benchmarks/translation_benchmark.py --compare \
  benchmarks/results/Qwen3.5-35B-baseline_ja_to_en.json \
  benchmarks/results/Qwen3.5-9B-FP8_ja_to_en.json

# 6. Stop 9B, restart 35B
echo ""
echo ">>> Stopping 9B, restarting 35B..."
docker stop scribe-translation-9b 2>/dev/null || true
docker rm scribe-translation-9b 2>/dev/null || true
docker compose -f docker-compose.gb10.yml up -d vllm-translation

echo ""
echo ">>> Waiting for 35B to load..."
for i in $(seq 1 30); do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8010/health 2>/dev/null)
  if [ "$status" = "200" ]; then
    echo ">>> 35B HEALTHY. Production restored."
    break
  fi
  sleep 10
done

echo ""
echo "=== BENCHMARK COMPLETE ==="
echo "Results in benchmarks/results/"
echo "Compare with: python benchmarks/translation_benchmark.py --compare <file_a> <file_b>"
