#!/usr/bin/env bash

    
python /workspace/wav2vec/run_common_voice.py \
    --model_name_or_path=$model_name_or_path \
    --dataset_config_name=$dataset_config_name \
    --output_dir=$output_dir \
    --cache_dir=$cache_dir \
    --overwrite_output_dir \
    --num_train_epochs=$num_train_epochs \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --evaluation_strategy=$evaluation_strategy \
    --learning_rate=$learning_rate \
    --warmup_steps=$warmup_steps \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps=$save_steps \
    --eval_steps=$eval_steps \
    --save_total_limit=$save_total_limit \
    --logging_steps=$logging_steps \
    --group_by_length \
    --feat_proj_dropout=$feat_proj_dropout \
    --layerdrop=$layerdrop \
    --gradient_checkpointing \
    --do_train \
    --do_eval \
    --max_train_samples $max_train_samples \
    --max_val_samples $max_val_samples \
    --report_to $report_to \
    --run_name $run_name \
    --augmentation_factor $augmentation_factor

