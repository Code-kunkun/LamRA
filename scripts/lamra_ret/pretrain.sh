NUM_GPUS=8
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_ID=qwen2_5-vl-7b                                  
TRAIN_DATA_PATH=./data/nli_for_simcse.csv  # path to the training data csv file
EVAL_DATA_PATH=None   

TRAIN_VISION_ENCODER=False                           
USE_VISION_LORA=False                                  
TRAIN_VISION_PROJECTOR=False                            

USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=64                                               
LORA_ALPHA=128                                           

RUN_ID=${MODEL_ID}_LamRA_Ret_Pretrain-Qwen2.5VL

DS_STAGE=zero2                                       
PER_DEVICE_BATCH_SIZE=72                               
GRAD_ACCUM=1                                           
NUM_EPOCHS=2                                            

LR=2e-4   # The training will be more stable under this learning rate
# LR=4e-4 # This learning rate may result in unstable training; consider multiple attempts, lowering it, or using our provided checkpoint
MODEL_MAX_LEN=1024                                  


torchrun $DISTRIBUTED_ARGS train/train_nli.py \
    --model_id $MODEL_ID \
    --data_path $TRAIN_DATA_PATH \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --use_flash_attn True
