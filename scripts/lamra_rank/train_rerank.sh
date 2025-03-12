NUM_GPUS=8
NPROC_PER_NODE=8
NNODES=2
NODE_RANK=1 # 1
MASTER_ADDR=`ip_address`
MASTER_PORT=29508

DISTRIBUTED_ARGS="
    --nnodes=${NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

# NUM_GPUS=8
# DISTRIBUTED_ARGS="
#     --nnodes=1 \
#     --nproc_per_node ${NUM_GPUS} \
#     --rdzv_backend c10d \
#     --rdzv_endpoint localhost:0
# "

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=qwen2_5-vl-7b                                 
MODEL_LOCAL_PATH=./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct
QUERY_DATA_PATH=./data/M-BEIR/query/union_train/mbeir_union_up_train.jsonl
CAND_POOL_PATH=./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl
INSTRUCTIONS_PATH=./data/M-BEIR/instructions/query_instructions.tsv
RERANK_DATA_PATH=./data/rerank_data_all.json
IMAGE_PATH_PREFIX=./data/M-BEIR

TRAIN_VISION_ENCODER=False                              
USE_VISION_LORA=False                                   
TRAIN_VISION_PROJECTOR=False                           

USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=128                                            
LORA_ALPHA=256                                   

RUN_ID=${MODEL_ID}_LamRA-Rank 

DS_STAGE=zero2                                         
PER_DEVICE_BATCH_SIZE=2                               
GRAD_ACCUM=4                                            
NUM_EPOCHS=1                                           

LR=2e-5                                                 
MODEL_MAX_LEN=2048                          


torchrun $DISTRIBUTED_ARGS train/train_rerank.py \
    --model_id $MODEL_ID \
    --query_data_path $QUERY_DATA_PATH \
    --cand_pool_path $CAND_POOL_PATH \
    --instructions_path $INSTRUCTIONS_PATH \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --model_local_path $MODEL_LOCAL_PATH \
    --rerank_data_path $RERANK_DATA_PATH \
    --image_path_prefix $IMAGE_PATH_PREFIX \
    --use_flash_attn True