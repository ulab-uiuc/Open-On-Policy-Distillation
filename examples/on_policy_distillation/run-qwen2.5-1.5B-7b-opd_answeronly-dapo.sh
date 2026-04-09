#!/bin/bash

# OPD-SGLang answer_only: external 8B teacher with privileged answer hint
# Training dataset: BytedTsinghua-SIA/DAPO-Math-17k
#
# Teacher mode (OPD-SGLang): answer_only
#   - Student: original problem, enable_thinking=False (no <think> in response)
#   - Teacher: same student user message + answer-only hint (privileged information)
#   - Distillation: reverse KL penalty via --opd-kl-coef
#
# Student and teacher both run with enable_thinking=false (chat template kwargs).
# Teacher prompt is built in slime.rollout.on_policy_distillation.reward_func.
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen2.5-1.5B-7b-opd_noanswer-dapo.sh
#   bash examples/on_policy_distillation/run-qwen2.5-1.5B-7b-opd_noanswer-dapo.sh \
#     --opd-kl-mode full_vocab_topk_reverse_kl --opd-topk 50

OPD_KL_MODE="token_reverse_kl"
OPD_TOPK="50"
OPD_EXPLICIT_LOSS_COEF="1.0"
OPD_DISTILL_MAX_RESPONSE_LEN="${OPD_DISTILL_MAX_RESPONSE_LEN:-2048}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --opd-kl-mode)
            OPD_KL_MODE="$2"
            shift 2
            ;;
        --opd-topk)
            OPD_TOPK="$2"
            shift 2
            ;;
        --opd-explicit-loss-coef)
            OPD_EXPLICIT_LOSS_COEF="$2"
            shift 2
            ;;
        --opd-distill-max-response-len)
            OPD_DISTILL_MAX_RESPONSE_LEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Supported args: --opd-kl-mode <token_reverse_kl|full_vocab_topk_reverse_kl> --opd-topk <int> --opd-explicit-loss-coef <float> --opd-distill-max-response-len <-1|int>"
            exit 1
            ;;
    esac
done

if [[ "${OPD_KL_MODE}" != "token_reverse_kl" && "${OPD_KL_MODE}" != "full_vocab_topk_reverse_kl" ]]; then
    echo "Invalid --opd-kl-mode: ${OPD_KL_MODE}"
    exit 1
fi
if ! [[ "${OPD_TOPK}" =~ ^[0-9]+$ ]] || [[ "${OPD_TOPK}" -le 0 ]]; then
    echo "Invalid --opd-topk: ${OPD_TOPK} (must be positive integer)"
    exit 1
fi
if ! [[ "${OPD_EXPLICIT_LOSS_COEF}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid --opd-explicit-loss-coef: ${OPD_EXPLICIT_LOSS_COEF} (must be numeric)"
    exit 1
fi
if ! [[ "${OPD_DISTILL_MAX_RESPONSE_LEN}" =~ ^-?[0-9]+$ ]] || { [[ "${OPD_DISTILL_MAX_RESPONSE_LEN}" -ne -1 ]] && [[ "${OPD_DISTILL_MAX_RESPONSE_LEN}" -le 0 ]]; }; then
    echo "Invalid --opd-distill-max-response-len: ${OPD_DISTILL_MAX_RESPONSE_LEN} (must be -1 or positive integer)"
    exit 1
fi
echo "OPD_KL_MODE=${OPD_KL_MODE}, OPD_TOPK=${OPD_TOPK}, OPD_EXPLICIT_LOSS_COEF=${OPD_EXPLICIT_LOSS_COEF}, OPD_DISTILL_MAX_RESPONSE_LEN=${OPD_DISTILL_MAX_RESPONSE_LEN}"

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONBUFFERED=16

###############################################################################
# Step -1: Start external SGLang teacher server (Qwen2.5-7B)
###############################################################################

TEACHER_IP="172.22.224.251"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-9}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.82}"
TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-2048}"
RM_MAX_CONCURRENCY="${RM_MAX_CONCURRENCY:-2}"
TEACHER_LOG_FILE="/tmp/sglang_teacher_qwen2.5_7b_$(date +%s).log"
TEACHER_STARTED_BY_SCRIPT=0

# curl "http://172.22.224.251:13141/health_generate"
echo "Starting teacher model server (model=${TEACHER_MODEL_PATH})..."
if curl -sf --max-time 2 "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; then
    echo "Teacher server already running at ${TEACHER_IP}:${TEACHER_PORT}, reusing."
else
    CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" python3 -m sglang.launch_server \
        --model-path "${TEACHER_MODEL_PATH}" \
        --host 0.0.0.0 \
        --port "${TEACHER_PORT}" \
        --tp 1 \
        --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
        --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
        > "${TEACHER_LOG_FILE}" 2>&1 &
    TEACHER_STARTED_BY_SCRIPT=1
    TEACHER_PID=$!
    echo "Teacher server pid=${TEACHER_PID}, log=${TEACHER_LOG_FILE}"

    MAX_WAIT_SECONDS=600
    ELAPSED=0
    until curl -sf --max-time 2 "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
        if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
            echo "ERROR: teacher server exited before ready."
            tail -n 80 "${TEACHER_LOG_FILE}" || true
            exit 1
        fi
        if [ "${ELAPSED}" -ge "${MAX_WAIT_SECONDS}" ]; then
            echo "ERROR: teacher server not ready after ${MAX_WAIT_SECONDS}s."
            tail -n 80 "${TEACHER_LOG_FILE}" || true
            exit 1
        fi
        echo "Waiting teacher server... (${ELAPSED}s/${MAX_WAIT_SECONDS}s)"
        tail -n 10 "${TEACHER_LOG_FILE}" || true
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done
fi
curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info" || true
echo "Teacher model server is ready at ${TEACHER_IP}:${TEACHER_PORT}."

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime_siqi/scripts/models/qwen2.5-1.5B.sh"

###############################################################################
# Step 0: Preprocess datasets
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset -------------------------------------------------------

TRAIN_DATASET="${TRAIN_DATASET:-BytedTsinghua-SIA/DAPO-Math-17k}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"   # DAPO does not require a config subset
TRAIN_OUT="/root/math/data/train_dapo.jsonl"

# EVAL datasets use boxed by default.
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"

TRAIN_ARGS=(--dataset "$TRAIN_DATASET" --split train --output "$TRAIN_OUT" --answer-format "$TRAIN_ANSWER_FORMAT")
[ -n "$TRAIN_CONFIG" ] && TRAIN_ARGS+=(--config "$TRAIN_CONFIG")
$PREPROCESS "${TRAIN_ARGS[@]}"


# ---- Eval datasets ----------------------------------------------------------
$PREPROCESS --dataset math-ai/aime24             --split test  --output /root/math/data/eval_aime24.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset math-ai/aime25             --split test  --output /root/math/data/eval_aime25.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset FlagEval/HMMT_2025         --split train --output /root/math/data/eval_hmmt.jsonl      --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset meituan-longcat/AMO-Bench  --split test  --output /root/math/data/eval_amo_bench.jsonl --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset HuggingFaceH4/MATH-500     --split test  --output /root/math/data/eval_math500.jsonl   --answer-format "$EVAL_ANSWER_FORMAT"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen2.5-1.5B
   --ref-load "/root/checkpoints_siqi/Qwen2.5-1.5B_torch_dist"
   --save /root/slime_siqi/output/Qwen2.5-1.5B_opd_sglang_answeronly_dapo/
   --save-interval 2000
)
# '''
# 是的，**如果你想“只用显式项”**，那就应该把 `--opd-kl-coef` 设成 `0`。

# 因为现在两条路都可能生效：

# 1. `opd-kl-coef`：通过 `adv := adv - coef * KL` 影响 PG。  
# 2. `opd-explicit-loss-coef`：通过 `loss += coef * opd_explicit_loss` 显式加到总 loss。

# 所以：

# - 只做显式蒸馏：`opd-explicit-loss-coef > 0` 且 `opd-kl-coef = 0`
# - 只做原 OPD（adv 惩罚）：`opd-explicit-loss-coef = 0` 且 `opd-kl-coef > 0`
# - 两者同时开：不是不行，但等于双重施压，通常需要重新调系数。
# '''
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1024}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.68}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-1024}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-examples/on_policy_distillation/eval_config_qwen2.5_1.5b_7b_safe.yaml}"

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_OUT"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --reward-key accuracy_strict
   --eval-reward-key accuracy_strict
   --rm-max-concurrency "${RM_MAX_CONCURRENCY}"
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-config "${EVAL_CONFIG_PATH}"
    --log-passrate
    # --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
   --opd-kl-mode "${OPD_KL_MODE}"
   --opd-topk "${OPD_TOPK}"
   --opd-explicit-loss-coef "${OPD_EXPLICIT_LOSS_COEF}"
   --opd-distill-max-response-len "${OPD_DISTILL_MAX_RESPONSE_LEN}"
   --opd-zero-task-reward
   --opd-teacher-info-mode answer_only
   --opd-teacher-tokenizer "${TEACHER_MODEL_PATH}"
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen2.5-1.5B-7b-opd-answeronly-dapo
   --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
)

if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
   SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=3,4,5,6
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "3,4,5,6"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

RAY_EXIT_CODE=$?
set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

####clear after training
if [ "${TEACHER_STARTED_BY_SCRIPT}" -eq 1 ]; then
    pkill -9 sglang || true
fi
ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python
