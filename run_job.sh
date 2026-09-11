#!/bin/bash
# Shared launcher; called by each model/experiment PBS file.
set -euo pipefail
MODEL=${1:?model required}; EXPERIMENT=${2:-all}
CODE_DIR=${CODE_DIR:-/lstr/sahara/datalab-ml/ibrahim/limagents_update/futureScope/exp_curr_without_extract}
cd "$CODE_DIR"
set +u
source activate image_lim_vllm_final
set -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/lstr/sahara/datalab-ml/ibrahim/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="127.0.0.1,localhost,::1,${no_proxy:-}"; export NO_PROXY="$no_proxy"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
unset VLLM_API_KEY
case "$MODEL" in
 gemma) DEFAULT_MODEL=/lstr/sahara/datalab-ml/ibrahim/models/gemma3_27b_it; QUANT=bnb;;
 qwen) DEFAULT_MODEL=/lstr/sahara/datalab-ml/ibrahim/models/qwen3_8b; QUANT=none;;
 mistral) DEFAULT_MODEL=/lstr/sahara/datalab-ml/ibrahim/models/mistral_small_3_1_24b_instruct; QUANT=bnb;;
 *) echo "Unknown model $MODEL"; exit 2;;
esac
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL}
DATA_DIR=${DATA_DIR:-/lstr/sahara/datalab-ml/ibrahim/limagents_update/futureScope/data}
NUM_SAMPLES=${NUM_SAMPLES:-10} # --num-samples: per venue/year, -1 means all.
RUN_ID=${RUN_ID:-${PBS_JOBID:-local}_$(date +%Y%m%dT%H%M%S)}
OUT_DIR=${OUT_DIR:-$CODE_DIR/results/$MODEL/$EXPERIMENT/$RUN_ID}
mkdir -p "$OUT_DIR"
# Prevent two processes from writing one resume directory simultaneously.
exec 9>"$OUT_DIR/.run.lock"
flock -n 9 || { echo "Another job owns $OUT_DIR"; exit 2; }
exec > >(tee -a "$OUT_DIR/job.txt") 2>&1
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
METHODS=${METHODS:-"nmf lda bertopic"}
read -r -a METHOD_ARRAY <<< "$METHODS"
PREFLIGHT_EXTRA=()
[[ "${NO_TOPIC_LABELS:-0}" == 1 ]] && PREFLIGHT_EXTRA+=(--no-topic-labels)
python preflight.py --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" --methods "${METHOD_ARRAY[@]}" --embedding-model "${EMBEDDING_MODEL:-/lstr/sahara/datalab-ml/ibrahim/models/all-MiniLM-L6-v2}" "${PREFLIGHT_EXTRA[@]}"
python -m unittest discover -s tests -v
VLLM_URL=http://127.0.0.1:8000/v1
if [[ "${NO_TOPIC_LABELS:-0}" != 1 && "$EXPERIMENT" != forecast ]]; then
TEMPLATE_ARGS=()
[[ -n "${CHAT_TEMPLATE:-}" ]] && TEMPLATE_ARGS+=(--template "$CHAT_TEMPLATE")
python chat_templates.py --model "$MODEL" --model-dir "$MODEL_DIR" --output "$OUT_DIR/chat_template.jinja" "${TEMPLATE_ARGS[@]}"
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
SERVER_ARGS=(--model "$MODEL_DIR" --served-model-name "$MODEL_DIR" --host 127.0.0.1 --port "$PORT"
 --dtype bfloat16 --max-model-len "$CONTEXT_LENGTH" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
 --max-num-seqs 2 --enforce-eager --seed 42 --chat-template "$OUT_DIR/chat_template.jinja")
if [[ "$QUANT" == bnb ]]; then SERVER_ARGS+=(--quantization bitsandbytes --load-format bitsandbytes); fi
python -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}" > "$OUT_DIR/vllm_server.log" 2>&1 &
VPID=$!
cleanup() { kill "$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python wait_server.py --url "http://127.0.0.1:$PORT" --pid "$VPID" --model "$MODEL_DIR" --timeout 1800
python schema_smoke.py --out-dir "$OUT_DIR" --model "$MODEL" --model-dir "$MODEL_DIR" \
 --chat-template "$OUT_DIR/chat_template.jinja" --vllm-url "http://127.0.0.1:$PORT/v1" \
 --context-length "$CONTEXT_LENGTH" --max-tokens "${MAX_TOKENS:-1024}" --chunk-tokens "${CHUNK_TOKENS:-6000}"
VLLM_URL="http://127.0.0.1:$PORT/v1"
fi
EXTRA=()
[[ -f "$OUT_DIR/chat_template.jinja" ]] && EXTRA+=(--chat-template "$OUT_DIR/chat_template.jinja")
[[ -n "${NEURIPS_YEAR_COLUMN:-}" ]] && EXTRA+=(--neurips-year-column "$NEURIPS_YEAR_COLUMN")
[[ "${NO_TOPIC_LABELS:-0}" == 1 ]] && EXTRA+=(--no-topic-labels)
python -u run_experiments.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --model "$MODEL" --model-dir "$MODEL_DIR" \
 --vllm-url "$VLLM_URL" --experiment "$EXPERIMENT" --num-samples "$NUM_SAMPLES" \
 --methods "${METHOD_ARRAY[@]}" --embedding-model "${EMBEDDING_MODEL:-/lstr/sahara/datalab-ml/ibrahim/models/all-MiniLM-L6-v2}" \
 --context-length "$CONTEXT_LENGTH" --chunk-tokens "${CHUNK_TOKENS:-6000}" --max-tokens "${MAX_TOKENS:-1024}" \
 --num-topics "${NUM_TOPICS:-20}" --activity-source "${ACTIVITY_SOURCE:-input}" "${EXTRA[@]}"
