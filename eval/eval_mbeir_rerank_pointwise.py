import os 
import sys
import numpy as np
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
import json 
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from dataset.datasets_mbeir_eval_rerank import LazySupervisedDataset
import torch 
from tqdm import tqdm 
from collators.eval_rerank import EvalRerankDataCollator
from torch.utils.data import DataLoader 
from accelerate import Accelerator
import accelerate 
import argparse 


def rerank(args):
    query_data_path = args.query_data_path 
    cand_pool_path = args.cand_pool_path 
    instructions_path = args.instructions_path
    model_id = args.model_id 
    original_model_id = args.original_model_id 
    ret_query_data_path = args.ret_query_data_path 
    ret_cand_data_path = args.ret_cand_data_path 
    rank_num = args.rank_num  
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer 
    tokenizer.padding_side = 'left'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True, 
    )
    model.eval()

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process 

    model = model.to(device)

    dataset = LazySupervisedDataset(query_data_path, cand_pool_path, instructions_path, ret_query_data_path, ret_cand_data_path, image_path_prefix=args.image_path_prefix, rank_num=rank_num)
    data_collator = EvalRerankDataCollator(tokenizer=tokenizer, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=data_collator)

    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    all_scores = []
    all_indexes = []

    dataloader, model = accelerator.prepare(dataloader, model)

    for inputs, indexes in tqdm(dataloader):
        inputs = tensors_to_device(inputs, device)
        outputs = model.generate(**inputs, max_new_tokens=128, output_scores=True, return_dict_in_generate=True, do_sample=False)
        generated_ids = outputs.sequences
        logits = outputs.scores[0] # (batch_size, vocab_size)
        scores = []
        for idx in range(len(logits)):
            probs = (torch.nn.functional.softmax(torch.FloatTensor([logits[idx][tokenizer("Yes").input_ids[0]], logits[idx][tokenizer("No").input_ids[0]]]), dim=0).detach().cpu().numpy())
            scores.append(probs[0])
        scores = accelerator.gather_for_metrics(scores)
        indexes = accelerator.gather_for_metrics(indexes)

        all_indexes.extend(indexes)
        all_scores.extend(scores)


    # reduce redundancy
    index_set = set()
    filter_indexes = []
    filter_scores = []

    if is_main_process:
        for idx, index in enumerate(all_indexes):
            if index in index_set:
                pass 
            else:
                index_set.add(index)
                filter_indexes.append(index)
                filter_scores.append(all_scores[idx])
        
        filter_indexes = np.array(filter_indexes) 
        sorted_filter_indices = np.argsort(filter_indexes)
        filter_scores = np.array(filter_scores)
        filter_scores = filter_scores[sorted_filter_indices]

        query_ids = []
        queryid2rerank_score = {}
        for item in dataset.query_data:
            query_ids.append(item['qid'])
        for i, query_id in enumerate(query_ids):
            if query_id not in queryid2rerank_score:
                queryid2rerank_score[query_id] = []
            for j in range(rank_num):
                queryid2rerank_score[query_id].append(float(filter_scores[i * rank_num + j]))


        save_dir_name = './mbeir_rerank_files'
        if not os.path.exists(save_dir_name):
            os.mkdir(save_dir_name)
        with open(f"{save_dir_name}/{args.save_name}_test_queryid2rerank_score.json", 'w') as f:
            json.dump(queryid2rerank_score, f, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data_path', type=str)
    parser.add_argument('--cand_pool_path', type=str)
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--ret_query_data_path', type=str)
    parser.add_argument('--ret_cand_data_path', type=str)
    parser.add_argument('--rank_num', type=int, default=50)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_path_prefix', type=str)

    args = parser.parse_args()
    rerank(args)