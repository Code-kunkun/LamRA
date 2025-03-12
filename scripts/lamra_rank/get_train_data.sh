ORIGINAL_MODEL_ID="./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID="./checkpoints/qwen2-vl-7b_LamRA_Ret"

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_visualnews_train_task0.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_visualnews_task0_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_visualnews_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name visualnews_task0

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_mscoco_train_task0.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_mscoco_task0_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_mscoco_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name mscoco_task0

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_fashion200k_train_task0.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_fashion200k_task0_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_fashion200k_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name fashion200k_task0

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_webqa_train_task1.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_webqa_task1_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_webqa_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name webqa_task1

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_edis_train.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_edis_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_edis_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name edis_task2

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_webqa_train_task2.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_webqa_task2_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_webqa_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name webqa_task2

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_visualnews_train_task3.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_visualnews_task3_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_visualnews_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name visualnews_task3

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_mscoco_train_task3.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_mscoco_task3_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_mscoco_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name mscoco_task3

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_fashion200k_train_task3.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_fashion200k_task3_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_fashion200k_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name fashion200k_task3

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_nights_train.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_nights_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_nights_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name nights_task4

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_oven_train_task6.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_oven_task6_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_oven_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name oven_task6

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_infoseek_train_task6.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_infoseek_task6_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_infoseek_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name infoseek_task6

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_fashioniq_train.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_fashioniq_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_fashioniq_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name fashioniq_task7

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_cirr_train.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_cirr_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_cirr_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name cirr_task7

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_oven_train_task8.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_oven_task8_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_oven_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name oven_task8

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 train/mbeir_get_rerank_train_data.py \
    --query_data_path ./data/M-BEIR/query/train/mbeir_infoseek_train_task8.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/rerank_pool/mbeir_infoseek_task8_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/train/mbeir_infoseek_train_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --model_id ${MODEL_ID} \
    --save_name infoseek_task8