#!/bin/bash

WORKSPACE=../
cd $WORKSPACE
export BENCHMARK_ROOT='./benchmark'

#export CUDA_VISIBLE_DEVICES=0
#export GPU=1
#export CUDA_DEVICE_ORDER=PCI_BUS_ID

work_dir=./results   
exp_name=math-llama-8b-dpts
trace_dir=./results/math-llama-8b/${exp_name}

mkdir -p ./results/math-llama-8b/${exp_name}

DATASET_NAME=math
MODEL_NAME=llama-8b
REWARD_MODEL=mistral_prm-7b

# Run with safe optimizations (no threading, no tensor dimension changes)
python3 main.py \
    --config configs/inference/DPTS.json \
    --work-dir $work_dir \
    --exp-name $exp_name \
    --data $DATASET_NAME \
    --model $MODEL_NAME \
    --reward_model $REWARD_MODEL \
    --trace-log-dir $trace_dir \
    --dtype bfloat16 \
    --fast_mode \
    --debug
