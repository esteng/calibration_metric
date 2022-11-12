#!/bin/bash 

CHECKPOINT_DIR=$1
# '/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000'
VALIDATION_FILE=examples/data/dev_medium.jsonl

python calibration_metric/examples/hf_generate.py \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${CHECKPOINT_DIR}/outputs_logits \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --get_logits 


