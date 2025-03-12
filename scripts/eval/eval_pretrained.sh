CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
    --image_data_path ./data/flickr/images \
    --text_data_path ./data/flickr/flickr_text.json \
    --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
    --model_id ./checkpoints/LamRA-Ret-Pretrained-Qwen2.5VL-merged

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_coco.py \
#     --image_data_path ./data/coco/coco_images \
#     --text_data_path ./data/coco/coco_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2_5-vl-7b_LamRA_Ret_Pretrain-Qwen2.5VL
