#!/bin/bash

# OPD-SGLang noanswer: external 8B teacher without privileged answer hint
# Training dataset: BytedTsinghua-SIA/DAPO-Math-17k
#
# Teacher mode (OPD-SGLang): same_as_student
#   - Student: original problem, enable_thinking=False (no <think> in response)
#   - Teacher: same student prompt, no privileged answer hint
#   - Distillation: reverse KL penalty via --opd-kl-coef
#
# Student and teacher both run with enable_thinking=false (chat template kwargs).
# Teacher prompt is built in slime.rollout.on_policy_distillation.reward_func.
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen3-1.7B-8b-opd_noanswer_dapo.sh
#   bash examples/on_policy_distillation/run-qwen3-1.7B-8b-opd_noanswer_dapo.sh \
#     --opd-kl-mode full_vocab_topk_reverse_kl --opd-topk 50

OPD_KL_MODE="topk_reverse_kl_notail_sg"
OPD_TOPK="5"
OPD_EXPLICIT_LOSS_COEF="1.0"
OPD_DISTILL_MAX_RESPONSE_LEN="${OPD_DISTILL_MAX_RESPONSE_LEN:-8192}"
OPD_TOKEN_STATS="${OPD_TOKEN_STATS:-1}"
OPD_TOKEN_STATS_TOPK="${OPD_TOKEN_STATS_TOPK:-5}"
OPD_TOKEN_STATS_REPEAT_NGRAM="${OPD_TOKEN_STATS_REPEAT_NGRAM:-3}"
OPD_TOKEN_STATS_EOS_TOKEN_ID="${OPD_TOKEN_STATS_EOS_TOKEN_ID:-151645}"
OPD_TEACHER_SFT="${OPD_TEACHER_SFT:-0}"
OPD_TEACHER_SFT_LOSS_COEF="${OPD_TEACHER_SFT_LOSS_COEF:-1.0}"
OPD_TEACHER_SFT_TEMPERATURE="${OPD_TEACHER_SFT_TEMPERATURE:-0.5}"
OPD_TEACHER_SFT_TOP_P="${OPD_TEACHER_SFT_TOP_P:-0.95}"
OPD_TEACHER_SFT_MAX_RESPONSE_LEN="${OPD_TEACHER_SFT_MAX_RESPONSE_LEN:-}"

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
        --opd-teacher-sft)
            OPD_TEACHER_SFT="1"
            shift
            ;;
        --opd-teacher-sft-loss-coef)
            OPD_TEACHER_SFT_LOSS_COEF="$2"
            shift 2
            ;;
        --opd-teacher-sft-temperature)
            OPD_TEACHER_SFT_TEMPERATURE="$2"
            shift 2
            ;;
        --opd-teacher-sft-top-p)
            OPD_TEACHER_SFT_TOP_P="$2"
            shift 2
            ;;
        --opd-teacher-sft-max-response-len)
            OPD_TEACHER_SFT_MAX_RESPONSE_LEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Supported args: --opd-kl-mode <token_reverse_kl|full_vocab_topk_reverse_kl|topk_reverse_kl_notail|topk_reverse_kl_notail_sg> --opd-topk <int> --opd-explicit-loss-coef <float> --opd-distill-max-response-len <-1|int> --opd-teacher-sft [--opd-teacher-sft-loss-coef <float>] [--opd-teacher-sft-temperature <float>] [--opd-teacher-sft-top-p <float>] [--opd-teacher-sft-max-response-len <int>]"
            exit 1
            ;;
    esac
done

if [[ "${OPD_KL_MODE}" != "token_reverse_kl" && \
      "${OPD_KL_MODE}" != "full_vocab_topk_reverse_kl" && \
      "${OPD_KL_MODE}" != "topk_reverse_kl_notail" && \
      "${OPD_KL_MODE}" != "topk_reverse_kl_notail_sg" ]]; then
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
if [[ "${OPD_TOKEN_STATS}" != "0" && "${OPD_TOKEN_STATS}" != "1" ]]; then
    echo "Invalid OPD_TOKEN_STATS: ${OPD_TOKEN_STATS} (must be 0 or 1)"
    exit 1
fi
if [[ "${OPD_TEACHER_SFT}" != "0" && "${OPD_TEACHER_SFT}" != "1" ]]; then
    echo "Invalid OPD_TEACHER_SFT: ${OPD_TEACHER_SFT} (must be 0 or 1)"
    exit 1
fi
if ! [[ "${OPD_TEACHER_SFT_LOSS_COEF}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid OPD_TEACHER_SFT_LOSS_COEF: ${OPD_TEACHER_SFT_LOSS_COEF} (must be numeric)"
    exit 1
fi
if ! [[ "${OPD_TEACHER_SFT_TEMPERATURE}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid OPD_TEACHER_SFT_TEMPERATURE: ${OPD_TEACHER_SFT_TEMPERATURE} (must be numeric)"
    exit 1
fi
if ! [[ "${OPD_TEACHER_SFT_TOP_P}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid OPD_TEACHER_SFT_TOP_P: ${OPD_TEACHER_SFT_TOP_P} (must be numeric)"
    exit 1
fi
if [[ -n "${OPD_TEACHER_SFT_MAX_RESPONSE_LEN}" ]] && { ! [[ "${OPD_TEACHER_SFT_MAX_RESPONSE_LEN}" =~ ^[0-9]+$ ]] || [[ "${OPD_TEACHER_SFT_MAX_RESPONSE_LEN}" -le 0 ]]; }; then
    echo "Invalid OPD_TEACHER_SFT_MAX_RESPONSE_LEN: ${OPD_TEACHER_SFT_MAX_RESPONSE_LEN} (must be positive integer)"
    exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_TOPK}" =~ ^[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_TOPK}" -le 0 ]]; then
    echo "Invalid OPD_TOKEN_STATS_TOPK: ${OPD_TOKEN_STATS_TOPK} (must be positive integer)"
    exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_REPEAT_NGRAM}" =~ ^[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_REPEAT_NGRAM}" -lt 2 ]]; then
    echo "Invalid OPD_TOKEN_STATS_REPEAT_NGRAM: ${OPD_TOKEN_STATS_REPEAT_NGRAM} (must be integer >= 2)"
    exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_EOS_TOKEN_ID}" =~ ^-?[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_EOS_TOKEN_ID}" -lt 0 ]]; then
    echo "Invalid OPD_TOKEN_STATS_EOS_TOKEN_ID: ${OPD_TOKEN_STATS_EOS_TOKEN_ID} (must be non-negative integer)"
    exit 1
fi
echo "OPD_KL_MODE=${OPD_KL_MODE}, OPD_TOPK=${OPD_TOPK}, OPD_EXPLICIT_LOSS_COEF=${OPD_EXPLICIT_LOSS_COEF}, OPD_DISTILL_MAX_RESPONSE_LEN=${OPD_DISTILL_MAX_RESPONSE_LEN}, OPD_TOKEN_STATS=${OPD_TOKEN_STATS}, OPD_TEACHER_SFT=${OPD_TEACHER_SFT}"

TOKEN_STATS_ARGS=()
if [[ "${OPD_TOKEN_STATS}" == "1" ]]; then
    echo "WARNING: OPD token stats are enabled. This adds teacher top-k/EOS diagnostics and can significantly slow rollout."
    TOKEN_STATS_ARGS+=(
        --opd-token-stats
        --opd-token-stats-topk "${OPD_TOKEN_STATS_TOPK}"
        --opd-token-stats-repeat-ngram "${OPD_TOKEN_STATS_REPEAT_NGRAM}"
        --opd-token-stats-eos-token-id "${OPD_TOKEN_STATS_EOS_TOKEN_ID}"
    )
fi

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONBUFFERED=16

###############################################################################
# Step -1: Start external SGLang teacher server (Qwen3-8B)
###############################################################################

TEACHER_IP="0.0.0.0"
TEACHER_PORT="${TEACHER_PORT:-30086}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-output/Qwen3-1.7B_opsd_masked_grpo_dapo_hf}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.70}"
RM_MAX_CONCURRENCY="${RM_MAX_CONCURRENCY:-64}"
TEACHER_LOG_FILE="/tmp/sglang_teacher_qwen3_8b_$(date +%s).log"
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
        --chunked-prefill-size 4096 \
        --watchdog-timeout 3600 \
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

source "/root/slime_siqi/scripts/models/qwen3-1.7B.sh"

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
   --hf-checkpoint Qwen/Qwen3-1.7B
   --ref-load "/root/checkpoints_siqi/Qwen3-1.7B_torch_dist"
   --save "${OPD_SAVE:-/root/slime_siqi/output/Qwen3-1.7B_8B_opd_noanswer_dapo/}"
   --save-interval 2000
)
if [[ -n "${OPD_LOAD:-}" ]]; then
   CKPT_ARGS+=(--load "${OPD_LOAD}")
fi
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
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-256}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-256}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-examples/on_policy_distillation/eval_config.yaml}"
OPD_TEACHER_SFT_MAX_RESPONSE_LEN="${OPD_TEACHER_SFT_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"

TEACHER_SFT_ARGS=()
if [[ "${OPD_TEACHER_SFT}" == "1" ]]; then
   echo "Teacher-generated online SFT is enabled: loss_coef=${OPD_TEACHER_SFT_LOSS_COEF}, temperature=${OPD_TEACHER_SFT_TEMPERATURE}, top_p=${OPD_TEACHER_SFT_TOP_P}, max_response_len=${OPD_TEACHER_SFT_MAX_RESPONSE_LEN}"
   TEACHER_SFT_ARGS+=(
      --opd-teacher-sft
      --opd-teacher-sft-loss-coef "${OPD_TEACHER_SFT_LOSS_COEF}"
      --opd-teacher-sft-temperature "${OPD_TEACHER_SFT_TEMPERATURE}"
      --opd-teacher-sft-top-p "${OPD_TEACHER_SFT_TOP_P}"
      --opd-teacher-sft-max-response-len "${OPD_TEACHER_SFT_MAX_RESPONSE_LEN}"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_OUT"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --rollout-top-p "${ROLLOUT_TOP_P}"
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
    --save-debug-rollout-data /root/slime_siqi/output/debug_rollout/{rollout_id}.pt
    --skip-eval-before-train
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
   --opd-teacher-info-mode same_as_student
   --opd-teacher-tokenizer "${TEACHER_MODEL_PATH}"
   ${TEACHER_SFT_ARGS[@]}
   ${TOKEN_STATS_ARGS[@]}
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-1.7B-1.7bgrpoteacher-opd-noanswer-dapo-nograd_reversekl_top5
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
export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 3 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

set +e
echo "Submitting Ray job..."
RAY_JOB_ID="qwen3_1p7b_8b_opd_noanswer_dapo_$(date +%s)"
ray job submit --submission-id "${RAY_JOB_ID}" --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "1,2,3"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
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
# if [ "${RAY_EXIT_CODE}" -eq 0 ]; then
#     echo "Submitted Ray job: ${RAY_JOB_ID}"
#     ray job logs --address="http://127.0.0.1:8265" -f "${RAY_JOB_ID}" || true
#     # while true; do
#     #     JOB_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_ID}" 2>&1)
#     #     echo "${JOB_STATUS_OUTPUT}"
#     #     if echo "${JOB_STATUS_OUTPUT}" | grep -q "Status: SUCCEEDED"; then
#     #         RAY_EXIT_CODE=0
#     #         break
#     #     fi
#     #     if echo "${JOB_STATUS_OUTPUT}" | grep -Eq "Status: (FAILED|STOPPED)"; then
#     #         RAY_EXIT_CODE=1
#     #         break
#     #     fi
#     #     sleep 15
#     # done
#     while true; do
#     JOB_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_ID}" 2>&1)
#     echo "${JOB_STATUS_OUTPUT}"

#     if echo "${JOB_STATUS_OUTPUT}" | grep -q ": SUCCEEDED"; then
#         RAY_EXIT_CODE=0
#         break
#     fi

#     if echo "${JOB_STATUS_OUTPUT}" | grep -Eq ": (FAILED|STOPPED)"; then
#         RAY_EXIT_CODE=1
#         break
#     fi

#     sleep 15
# done
# fi
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
