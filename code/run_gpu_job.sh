#!/bin/bash

OUTPUT_DIR="/home/ywang27/TSFSER/code/log"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

srun --account=pi-graziul \
     --job-name="train-ser" \
     --output="$OUTPUT_DIR/susas_cnn.stdout" \
     --error="$OUTPUT_DIR/susas_cnn.stderr" \
     --partition=gpu \
     --nodes=1 \
     --gpus=1 \
     --ntasks=1 \
     --gpus-per-task=1 \
     --mem-per-cpu=24G \
     --time=02:00:00 \
     python -u main.py