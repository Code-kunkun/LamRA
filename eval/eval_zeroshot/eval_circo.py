import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
import torch 
import argparse
from dataset.datasets_circo import CIRCODataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
import numpy as np 
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

    query_dataset = CIRCODataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type='query',
        split=args.split
    )

    cand_dataset = CIRCODataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type='image',
        split=args.split 
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

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)

        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        # Adjust the order according to ids 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        query_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        candidate_features = candidate_features[sorted_candidate_indices]

        ap_at5, ap_at10, ap_at25, ap_at50 = [], [], [], []
        precision_at5, precision_at10, precision_at25, precision_at50 = [], [], [], []
        recall_at5, recall_at10, recall_at25, recall_at50 = [], [], [], []

        annotations = query_dataset.annotations 
        max_num_gts = query_dataset.max_num_gts
        index_names = [str(item) for item in query_dataset.img_ids]
        assert len(annotations) == len(query_features)

        if args.split == 'val':

            for index in range(len(query_features)):
                target_img_id = str(annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in annotations[index]['gt_img_ids']]
                gt_img_ids += [''] * (max_num_gts - len(gt_img_ids))
                gt_img_ids = np.array(gt_img_ids)[np.array(gt_img_ids) != '']
                score = query_features[index] @ candidate_features.T 
                sorted_indices = torch.topk(score, dim=-1, k=50).indices.cpu()
                sorted_index_names = np.array(index_names)[sorted_indices]
                map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
                precisions = torch.cumsum(map_labels, dim=0) * map_labels

                precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)
                ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
                ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
                ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
                ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

                assert target_img_id == gt_img_ids[0], f"Target name not in GTs {target_img_id} {gt_img_ids}"
                single_gt_labels = torch.tensor(sorted_index_names == target_img_id)
                recall_at5.append(float(torch.sum(single_gt_labels[:5])))
                recall_at10.append(float(torch.sum(single_gt_labels[:10])))
                recall_at25.append(float(torch.sum(single_gt_labels[:25])))
                recall_at50.append(float(torch.sum(single_gt_labels[:50])))

            map_at5 = np.mean(ap_at5) * 100
            map_at10 = np.mean(ap_at10) * 100
            map_at25 = np.mean(ap_at25) * 100
            map_at50 = np.mean(ap_at50) * 100
            recall_at5 = np.mean(recall_at5) * 100
            recall_at10 = np.mean(recall_at10) * 100
            recall_at25 = np.mean(recall_at25) * 100
            recall_at50 = np.mean(recall_at50) * 100

            print('map_at5: ', map_at5)
            print('map_at10: ', map_at10)
            print('map_at25: ', map_at25)
            print('map_at50: ', map_at50)

        elif args.split == 'test':
            res = {}

            for index in range(len(query_features)):
                score = query_features[index] @ candidate_features.T 
                sorted_indices = torch.topk(score, dim=-1, k=50).indices.cpu()
                sorted_index_names = np.array(index_names)[sorted_indices]
                sorted_index_names = sorted_index_names.tolist()
                res[annotations[index]['id']] = sorted_index_names

            save_dir_name = "./zeroshot_retrieval_eval_results"

            if not os.path.exists(save_dir_name):
                os.mkdir(save_dir_name)

            with open(f"{save_dir_name}/circo_test_retrieval_results.json", 'w') as f:
                json.dump(res, f)

            # save for rerank
            if args.save_for_rerank:
                save_for_rerank(query_features, candidate_features, query_ids, index_names, save_dir_name)

def save_for_rerank(query_features, candidate_features, query_ids, index_names, save_dir_name):
    index = []
    scores = []
    for i in range(len(query_features)):
        score = query_features[i] @ candidate_features.T 
        topk_score, topk_indexes = torch.topk(score, k=100, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        index.append(topk_indexes)
        scores.append(topk_score.tolist())
    
    cand_names = np.array([[index_names[item] for item in row] for row in index])
    query_names = query_ids 

    with open(f"{save_dir_name}/circo_test_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/circo_test_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/circo_test_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path_prefix', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_for_rerank', action='store_true')

    args = parser.parse_args()
    eval(args)