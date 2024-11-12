#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python ../src/train.py \
    --template falcon \
    --stage sft \
    --model_name_or_path ../models/Falcon_11b \
    --do_predict \
    --dataset ed_tst \
    --dataset_dir ../data/ed_json_data \
    --dataset ed_tst \
    --eval_dataset ed_tst \
    --output_dir output_ed/ \
    --overwrite_cache \
    --preprocessing_num_workers 2 \
    --per_device_eval_batch_size 20 \
    --predict_with_generate 

python ed_scripts/cal_metrics.py

#    --template mistral \
#    --stage sft \
#    --model_name_or_path ../models/Mistral_Nemo \


#    --template phi \
#    --stage sft \
#    --model_name_or_path ../models/Phi_35 \

#    --template chatglm3 \
#    --stage sft \
#    --model_name_or_path ../models/ChatGLM3 \



#CUDA_VISIBLE_DEVICES=$1 python ../src/train_bash.py \
#    --stage sft \
#    --model_name_or_path ../models/ChatGLM3 \
#    --do_train \
#    --dataset ed_tra \
#    --dataset_dir ../data/ed_json_data \
#    --finetuning_type lora \
#    --output_dir checkpoint/ed_checkpoint_sit_cs \
#    --overwrite_cache \
#    --per_device_train_batch_size 12 \
#    --gradient_accumulation_steps 2 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --warmup_steps 0 \
#    --learning_rate 1e-3 \
#    --num_train_epochs 3.0 \
#    --quantization_bit 8 \
#    --fp16

