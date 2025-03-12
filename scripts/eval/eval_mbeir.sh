MODEL_ID="./checkpoints/qwen2_5-vl-7b_LamRA-Ret-Qwen2.5VL"
ORIGINAL_MODEL_ID=./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct 
IMAGE_PATH_PREFIX=./data/M-BEIR

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_mscoco_task0_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_mscoco_task0_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_mscoco_task3_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_cirr_task7_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_cirr_task7_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_fashioniq_task7_test.jsonl \
#     --query_cand_pool ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_fashioniq_task7_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_webqa_task1_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_webqa_task1_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_nights_task4_test.jsonl \
#     --query_cand_pool ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_nights_task4_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_nights_task4_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_oven_task6_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_oven_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
    --query_data_path ./data/M-BEIR/query/test/mbeir_infoseek_task6_test.jsonl \
    --query_cand_pool ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/test/mbeir_infoseek_task6_test_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --image_path_prefix ${IMAGE_PATH_PREFIX} \
    --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_fashion200k_task0_test.jsonl \
#     --query_cand_pool ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_fashion200k_task0_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_visualnews_task0_test.jsonl \
#     --query_cand_pool ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_visualnews_task0_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_webqa_task2_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_webqa_task2_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
    --query_data_path ./data/M-BEIR/query/test/mbeir_oven_task8_test.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/test/mbeir_oven_task8_test_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --image_path_prefix ${IMAGE_PATH_PREFIX} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
    --query_data_path ./data/M-BEIR/query/test/mbeir_infoseek_task8_test.jsonl \
    --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --qrels_path ./data/M-BEIR/qrels/test/mbeir_infoseek_task8_test_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --image_path_prefix ${IMAGE_PATH_PREFIX} \
    --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_fashion200k_task3_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_fashion200k_task3_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_visualnews_task3_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_visualnews_task3_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ./data/M-BEIR/query/test/mbeir_edis_task2_test.jsonl \
#     --query_cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ./data/M-BEIR/cand_pool/local/mbeir_edis_task2_cand_pool.jsonl \
#     --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
#     --qrels_path ./data/M-BEIR/qrels/test/mbeir_edis_task2_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID}
