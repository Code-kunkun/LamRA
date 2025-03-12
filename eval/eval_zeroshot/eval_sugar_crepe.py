import os 
import json
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
import torch 
import argparse
from dataset.datasets_sugar_crepe import SugarCrepeDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate


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

    query_dataset = SugarCrepeDataset(
        annotation_path_prefix=args.annotation_path_prefix,
        data_type=args.data_type, 
        image_path_prefix=args.image_path_prefix, 
        type='image',
    )

    cand_dataset = SugarCrepeDataset(
        annotation_path_prefix=args.annotation_path_prefix,
        data_type=args.data_type, 
        image_path_prefix=args.image_path_prefix, 
        type='text',
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
        # Adjust the order according to ids 
        import numpy as np 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        image_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        text_features = candidate_features[sorted_candidate_indices]

        # texts_image_index = [i for i in range(image_features.shape[0])]
        scores = image_features @ text_features.T # (N, 2N)
        hit = 0
        
        for i in range(len(scores)):
            if scores[i][i * 2] > scores[i][i * 2 + 1]:
                hit += 1
        print('accuracy: ', hit / len(scores))

        save_dir_name = "./zeroshot_retrieval_eval_results"

        model_name = args.model_id.split('/')[-1]
        with open(f"{save_dir_name}/{model_name}.txt", 'a') as f:
            f.write(f'sugar crepe {args.data_type} evaluation' + '\n')
            f.write(f"accuracy = {hit / len(scores)}" + "\n")

        # save_for_rerank
        if args.save_for_rerank:
            save_for_rerank(scores, args.data_type, save_dir_name)

def save_for_rerank(scores, data_type, save_dir_name):
    save_score = []
    for i in range(len(scores)):
        for j in range(2):
            save_score.append(scores[i][i * 2 + j].item())
    with open(f"{save_dir_name}/sugar_crepe_{data_type}.json", 'w') as f:
        json.dump(save_score, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--annotation_path_prefix', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_for_rerank', action='store_true')

    args = parser.parse_args()
    eval(args)