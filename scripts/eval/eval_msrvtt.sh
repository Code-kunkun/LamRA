CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msrvtt.py \
    --data_path ./data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv \
    --video_data_path ./data/msrvtt/videos \
    --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --model_id code-kunkun/LamRA-Ret
