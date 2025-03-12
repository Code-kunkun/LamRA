CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msvd.py \
    --video_path_prefix ./data/MSVD/YouTubeClips \
    --test_video_path ./data/MSVD/test_list.txt \
    --captions_path ./data/MSVD/raw-captions.pkl \
    --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --model_id code-kunkun/LamRA-Ret