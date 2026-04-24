#!/bin/bash

# Privileged GRPO: GRPO training where the rollout prompt already contains
# privileged information (reference solution / answer hint).
#
# How it works
# ------------
# 1. Preprocess: filter_openthoughts_math.py --privileged-mode embeds the
#    privileged information directly into the "prompt" field of the training JSONL.
# 2. Rollout: the model generates responses conditioned on the privileged prompt
#    (fully on-policy -- no token replacement after the fact).
# 3. Reward: standard math correctness reward.
# 4. Training: standard GRPO on (privileged_prompt, response) pairs.
#    No teacher forward pass, no KL loss.
#
# Privileged modes (PRIVILEGED_MODE env var):
#   full        (default): full reference solution prepended with re-derivation instruction
#   answer_only : "The correct final answer is X. Now solve it yourself."
#   conciseness : conciseness instruction prepended (no ground truth needed)
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen3-1.7B-opsd_grpo-openthoughts_privileged_grpo.sh
#
# Override mode:
#   PRIVILEGED_MODE=answer_only bash examples/on_policy_distillation/run-qwen3-1.7B-opsd_grpo-openthoughts_privileged_grpo.sh

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime_siqi/scripts/models/qwen3-1.7B.sh"

###############################################################################
# Config
###############################################################################

PRIVILEGED_MODE="${PRIVILEGED_MODE:-full}"   # full | answer_only | conciseness

###############################################################################
# Step 0: Preprocess datasets
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset (privileged prompt baked in at preprocessing time) ----
TRAIN_OUT="/root/math/data/train_openthoughts_math_privileged_${PRIVILEGED_MODE}.jsonl"
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-1234}"

python3 examples/on_policy_distillation/filter_openthoughts_math.py \
    --output "$TRAIN_OUT" \
    --answer-format "$TRAIN_ANSWER_FORMAT" \
    --privileged-mode "$PRIVILEGED_MODE"

# ---- Eval datasets (standard prompts, no privileged info) -------------------
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
   --save "/root/slime_siqi/output/Qwen3-1.7B_privileged_grpo_${PRIVILEGED_MODE}_openthoughts/"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_OUT"
   --input-key prompt
   --label-key label
   --rollout-seed "${ROLLOUT_SEED}"
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --over-sampling-batch-size 16
   --rollout-max-prompt-len 24576
   --global-batch-size 16
   --balance-data
)

# post_process_rewards_grpo_only: pure GRPO normalisation, no teacher forward pass.
# The privileged information is already in sample.prompt (baked in at preprocess time).
RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards_grpo_only
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --eval-temperature 0.6
    --log-passrate
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
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group "qwen3-1.7B-privileged_grpo_${PRIVILEGED_MODE}-openthoughts-nothinking"
   --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.78
)

MISC_ARGS=(
   --seed "${EXPERIMENT_SEED}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --use-fault-tolerance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=4,5,6
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 3 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "4,5,6"
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
set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

####clear after training
ray stop --force
