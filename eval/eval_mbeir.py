import json
from transformers import AutoProcessor
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
import torch 
import argparse
from dataset.datasets_mbeir import QueryDataset, CandidateDataset
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

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
    # tokenizer.model_max_length = args.model_max_length

    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens

        model.resize_token_embeddings(len(tokenizer))

        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids

    add_embed_token(tokenizer, model)

    query_dataset = QueryDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    cand_dataset = CandidateDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    query_data_collator = MbeirQueryDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = MbeirCandidateDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

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

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, _, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)

        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids, _ = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        # Adjust the order according to ids 
        import numpy as np 

        index = []
        scores = []
        for i in range(len(query_features)):
            query_feature = query_features[i:i+1]
            score = query_feature @ candidate_features.T # (1, num_candidate)
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)
            scores.append(topk_score.tolist())

        cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
        query_names = [unhash_qid(item) for item in query_ids]


        save_dir_name = "./LamRA_Ret_eval_results"
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_name = args.qrels_path.split('/')[-1].replace('_qrels.txt', '')
        model_name = args.model_id.split('/')[-1]
        save_name = f"{save_name}_{model_name}"
        with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
            json.dump(query_names, f, indent=2)
        with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
            json.dump(cand_names.tolist(), f, indent=2)
        with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
            json.dump(scores, f, indent=2)
        torch.save(query_features.cpu(), f"{save_dir_name}/{save_name}_query_features.pth")
        torch.save(candidate_features.cpu(), f"{save_dir_name}/{save_name}_candidate_features.pth")
        with open(f"{save_dir_name}/{save_name}_query_ids.json", 'w') as f:
            json.dump(query_ids, f, indent=2)
        with open(f"{save_dir_name}/{save_name}_candidate_ids.json", 'w') as f:
            json.dump(candidate_ids, f, indent=2)
        

        qrel, qid_to_taskid = load_qrel(args.qrels_path)

        k_lists = [1, 5, 10, 50]
        res = {}

        for k in k_lists:
            res[f'recall_{k}'] = []

        for ind, query_name in enumerate(tqdm(query_names)):
            relevant_docs = qrel[query_name]
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)

        for k in k_lists:
            print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

        model_name = model_id.split('/')[-1]
        with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
            f.write(args.qrels_path + '\n')
            for k in k_lists:
                f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data_path', type=str)
    parser.add_argument('--cand_pool_path', type=str)
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--qrels_path', type=str)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--query_cand_pool_path', type=str)
    parser.add_argument('--image_path_prefix', type=str)

    args = parser.parse_args()
    eval(args)