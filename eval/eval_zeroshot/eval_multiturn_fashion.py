import os 
import numpy as np
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
import torch 
import argparse
from dataset.datasets_multiturn_fashion import MultiTurnFashionDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
import json 


def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True, 
    )

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer 
    tokenizer.padding_side = 'left'

    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens

        model.resize_token_embeddings(len(tokenizer))

        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids

    add_embed_token(tokenizer, model)

    query_dataset = MultiTurnFashionDataset(
        query_data_path=args.query_data_path, 
        cand_data_path=args.cand_data_path,
        image_path_prefix=args.image_path_prefix,
        type='query',
        category=args.category
    )

    cand_dataset = MultiTurnFashionDataset(
        query_data_path=args.query_data_path, 
        cand_data_path=args.cand_data_path,
        image_path_prefix=args.image_path_prefix,
        type='cand',
        category=args.category
    )

    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    from tqdm import tqdm 
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)

        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        import numpy as np 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        query_features = query_features[sorted_query_indices]

        # get the score for each text and image pair
        index = []
        for i in range(len(query_features)):
            query_feature = query_features[i:i+1]
            score = query_feature @ candidate_features.T 
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)

        cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
        query_names = query_ids 

        k_lists = [5, 8, 10]
        res = {}

        for k in k_lists:
            res[f"recall_{k}"] = []

        for ind, query_name in enumerate(tqdm(query_names)):
            relevant_docs = [query_dataset.query_data[query_name]['target'][1]]
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)

        for k in k_lists:
            print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

        model_name = args.model_id.split('/')[-1]
        save_dir_name = "./zeroshot_retrieval_eval_results"
        
        if not os.path.exists(save_dir_name):
            os.mkdir(save_dir_name)

        with open(f"{save_dir_name}/{model_name}.txt", 'a') as f:
            f.write(f'multi-turn fashion retrieval {args.category} evaluation' + '\n')
            for key in res:
                f.write(f"{key} = {sum(res[key]) / len(res[key])}" + "\n")

        if args.save_for_rerank:
            save_for_rerank(query_features, candidate_features, query_ids, candidate_ids, save_dir_name)

def save_for_rerank(query_features, candidate_features, query_ids, candidate_ids, save_dir_name):
    index = []
    scores = []
    for i in range(len(query_features)):
        score = query_features[i : i + 1] @ candidate_features.T 
        topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        index.append(topk_indexes)
        scores.append(topk_score.tolist())
    
    cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
    query_names = query_ids 

    with open(f"{save_dir_name}/mrf_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/mrf_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/mrf_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents

    # Get the set of indices for the top k retrieved documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])

    # Convert the relevant documents to a set
    relevant_docs_set = set(relevant_docs)

    # Check if there is an intersection between relevant docs and top k retrieved docs
    # If there is, we return 1, indicating successful retrieval; otherwise, we return 0
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data_path', type=str)
    parser.add_argument('--cand_data_path', type=str)
    parser.add_argument('--category', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--save_for_rerank', action='store_true')

    args = parser.parse_args()
    eval(args)