#!/bin/bash
# The PaddleOCR stack behind ocralign's paddle backend (process_pdf(..., backend="paddle")).
# Run it as `python -m ocralign.backends.paddle up|down|status`, or directly:
#
#   bash stack.sh up        start both services, wait until healthy
#   bash stack.sh down      stop both — frees the GPU
#   bash stack.sh status    container state, health, GPU holders
#
#   :8080  paddleocr_server     vLLM serving PaddleOCR-VL-1.6-0.9B (recognition)
#   :8081  paddleocr_pipeline   service.py beside this script (layout detection + page assembly)
#
# GPU: the stack needs ~13 GB FREE to start (vLLM reserves gpu-memory-utilization 0.5 of the
# card, see backend_config.yaml) and holds ~9.3 GB once running. On a 22-24 GB card it cannot run
# beside another large model (e.g. a 20B LLM), so work in two phases: OCR with the stack up and
# keep the payloads (fetch_payload), then `down` and start the other model.
#
# Both services bind 127.0.0.1 only: documents may be PHI and a host may have a public address.
# Everything is env-overridable. The client image is built locally from Dockerfile.client beside
# this script; `up` prints the build command if it is missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VL_NAME="${OCR_VL_CONTAINER:-paddleocr_server}"
SVC_NAME="${OCR_SVC_CONTAINER:-paddleocr_pipeline}"
VL_IMAGE="${OCR_VL_IMAGE:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu}"
CLIENT_IMAGE="${OCR_CLIENT_IMAGE:-paddleocr-vl-client:3.6.1}"
PADDLE_CACHE="${PADDLE_CACHE:-$HOME/paddle-cache}"     # model weights; downloaded once
VL_PORT="${OCR_VL_PORT:-8080}"
SVC_PORT="${OCR_SVC_PORT:-8081}"
GPU_DEVICE="${OCR_GPU_DEVICE:-0}"
VL_GPU_MIB="${OCR_VL_GPU_MIB:-11500}"     # vLLM reserves gpu-memory-utilization 0.5 of 22 GB up front
SVC_GPU_MIB="${OCR_SVC_GPU_MIB:-1500}"    # PP-DocLayoutV3 on paddle-gpu

running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
exists()  { docker ps -a --format '{{.Names}}' | grep -qx "$1"; }

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_DEVICE" | tr -d ' '
}

require_gpu() {  # $1 = MiB needed, $2 = what for
  local free
  free="$(gpu_free_mib)"
  if (( free < $1 )) && [[ "${OCR_FORCE:-0}" != "1" ]]; then
    echo "not enough free GPU for $2: ${free} MiB free, needs ~$1 MiB. Current holders:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    echo "stop the LLM first (they cannot share the A10G), or OCR_FORCE=1 to try anyway." >&2
    exit 1
  fi
}

require_image() {  # the client image is built locally, never pulled
  if ! docker image inspect "$CLIENT_IMAGE" >/dev/null 2>&1; then
    echo "image $CLIENT_IMAGE not found. Build it once (it layers on the VL server image):" >&2
    echo "  docker build -t \"$CLIENT_IMAGE\" -f \"$HERE/Dockerfile.client\" \"$HERE\"" >&2
    exit 1
  fi
}

wait_healthy() {  # $1 = url, $2 = name, $3 = timeout seconds
  local deadline=$(( $(date +%s) + ${3:-300} ))
  printf "waiting for %s " "$2"
  until curl -sf -o /dev/null "$1"; do
    if (( $(date +%s) > deadline )); then
      echo; echo "$2 did not become healthy within ${3:-300}s — last log lines:" >&2
      docker logs --tail 20 "$2" >&2 || true
      exit 1
    fi
    printf "."; sleep 3
  done
  echo " ok"
}

up() {
  require_image
  # Create the weight cache before Docker does. A bind-mount source that does not exist is
  # created by the daemon as root, while the container runs as uid 1000 (paddleocr) and cannot
  # write to it — the model download then fails and `up` just times out with no obvious cause.
  mkdir -p "$PADDLE_CACHE"
  if ! running "$VL_NAME"; then
    require_gpu "$(( VL_GPU_MIB + SVC_GPU_MIB ))" "the OCR stack"
    if exists "$VL_NAME"; then
      docker start "$VL_NAME" >/dev/null
    else
      docker run -d --name "$VL_NAME" --gpus "device=$GPU_DEVICE" --network host --shm-size=8g \
        --restart unless-stopped --log-opt max-size=100m --log-opt max-file=3 \
        -v "$HERE/backend_config.yaml:/backend_config.yaml:ro" \
        -v "$PADDLE_CACHE:/home/paddleocr/.paddlex" \
        "$VL_IMAGE" \
        paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B \
          --host 127.0.0.1 --port "$VL_PORT" --backend vllm --backend_config /backend_config.yaml \
        >/dev/null
    fi
  fi
  wait_healthy "http://127.0.0.1:$VL_PORT/health" "$VL_NAME" 400

  if ! running "$SVC_NAME"; then
    require_gpu "$SVC_GPU_MIB" "the layout service"
    # Stateless, and service.py is bind-mounted: always recreate so the running code and env are
    # the checked-in ones, never a stale container from an older revision.
    exists "$SVC_NAME" && docker rm -f "$SVC_NAME" >/dev/null
    docker run -d --name "$SVC_NAME" --gpus "device=$GPU_DEVICE" --network host --shm-size=2g \
      --restart unless-stopped --log-opt max-size=50m --log-opt max-file=3 \
      -e VL_URL="http://127.0.0.1:$VL_PORT/v1" \
      -e OCR_SERVICE_HOST=127.0.0.1 -e OCR_SERVICE_PORT="$SVC_PORT" \
      -v "$HERE/service.py:/opt/paddle_ocr/service.py:ro" \
      -v "$PADDLE_CACHE:/home/paddleocr/.paddlex" \
      "$CLIENT_IMAGE" python /opt/paddle_ocr/service.py >/dev/null
  fi
  wait_healthy "http://127.0.0.1:$SVC_PORT/health" "$SVC_NAME" 300
  echo "OCR stack up: service http://127.0.0.1:$SVC_PORT (ocralign backend=\"paddle\")"
}

down() {
  # Service first: it depends on the VL server, not the other way round.
  for n in "$SVC_NAME" "$VL_NAME"; do
    if running "$n"; then docker stop "$n" >/dev/null && echo "stopped $n"; fi
  done
  echo "GPU free: $(gpu_free_mib) MiB"
}

status() {
  for pair in "$VL_NAME:$VL_PORT" "$SVC_NAME:$SVC_PORT"; do
    local n="${pair%%:*}" p="${pair##*:}" state health
    state="$(docker ps -a --filter "name=^${n}$" --format '{{.Status}}')"
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health")" || health="down"
    printf "  %-20s :%-5s %-28s health=%s\n" "$n" "$p" "${state:-absent}" "$health"
  done
  echo "  GPU free: $(gpu_free_mib) MiB"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /' || true
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
