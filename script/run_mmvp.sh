#!/bin/bash
set -euo pipefail
# Reproduce ULR results on MMVP.
# Usage: bash script/run_mmvp.sh [N]
# Env overrides: PYBIN, CUDA_VISIBLE_DEVICES, NUM_WORKERS, KV_REUSE(0/1)

export HF_HOME=.cache

N=${1:-300}
DATASET=mmvp
DATA_FILE=data/${DATASET}.json

PYBIN=${PYBIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
KV_REUSE=${KV_REUSE:-1}
KVFLAG=""; [ "$KV_REUSE" = "1" ] && KVFLAG="--kv_reuse"
TAG=$([ "$KV_REUSE" = "1" ] && echo kv || echo base)
OUT=./output/${DATASET}/n${N}_${TAG}

echo ">>> ${DATASET}  N=${N}  KV_REUSE=${KV_REUSE}  GPU=${CUDA_VISIBLE_DEVICES}  -> ${OUT}"

$PYBIN main.py \
    --dataset "$DATA_FILE" \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir "$OUT" \
    --device cuda --seed 42 \
    --max_new_tokens 2048 \
    --max_num_steps 20 --align_sgd_pre_steps 5 \
    --num_thought_tokens 4 --sigma 25 --sigma_decay 0.95 --lr 0.001 \
    --verbose 0 --min_pixels 128 --max_pixels 256 \
    --start_data_idx 0 --end_data_idx "$N" \
    --use_llm_verify \
    --num_workers "$NUM_WORKERS" --worker_device_round_robin \
    --align_pos_k 4 \
    --patch_select_from_question --align_contrastive \
    --align_contrastive_tau 0.2 --align_neg_k 4 --align_weight 1 \
    --question_patch_assign_mode index_slice \
    --solver_prompt_idx 5 --thought_vision_attn_reward_weight 0.05 \
    $KVFLAG
