import os
import sys 
import numpy as np

import json 
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from dataset.datasets_urban1k import UrbanRerankI2TDataset, UrbanRerankT2IDataset
from dataset.datasets_sharegpt4v import ShareGPT4VI2TDataset, ShareGPT4VT2IDataset
from dataset.datasets_flickr import FlickrRerankI2TDataset, FlickrRerankT2IDataset
from dataset.datasets_genecis import GenecisCOCORerankDataset, GenecisVAWRerankDataset
from dataset.datasets_ccneg import CCNegRerankDataset
from dataset.datasets_sugar_crepe import SugarCrepeRerankDataset
from dataset.datasets_circo import CIRCORerankDataset
from dataset.datasets_vist import VistRerankDataset
from dataset.datasets_visdial import VisDialRerankDataset
from dataset.datasets_multiturn_fashion import MultiTurnFashionRerankDataset
from dataset.datasets_msrvtt import MSRVTTRerankT2VDataset
from dataset.datasets_msvd import MSVDRerankT2VDataset
import torch 
from tqdm import tqdm 
from collators.eval_rerank import EvalRerankDataCollator
from torch.utils.data import DataLoader 
from accelerate import Accelerator
import argparse 


def rerank(args):
    model_id = args.model_id 
    original_model_id = args.original_model_id 
    ret_query_data_path = args.ret_query_data_path 
    ret_cand_data_path = args.ret_cand_data_path
    image_data_path = args.image_data_path
    text_data_path = args.text_data_path  
    annotation_path = args.annotation_path 
    image_path_prefix = args.image_path_prefix 
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

    if args.save_name.startswith('urban1k_t2i'):
        dataset = UrbanRerankT2IDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('urban1k_i2t'):
        dataset = UrbanRerankI2TDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('sharegpt4v_t2i'):
        dataset = ShareGPT4VT2IDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('sharegpt4v_i2t'):
        dataset = ShareGPT4VI2TDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('msrvtt'):
        dataset = MSRVTTRerankT2VDataset(ret_query_data_path, ret_cand_data_path, rank_num)
    elif args.save_name.startswith('msvd'):
        dataset = MSVDRerankT2VDataset(ret_query_data_path, ret_cand_data_path, rank_num)
    elif args.save_name.startswith('flickr_t2i'):
        dataset = FlickrRerankT2IDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('flickr_i2t'):
        dataset = FlickrRerankI2TDataset(image_data_path, text_data_path, ret_query_data_path, ret_cand_data_path, rank_num=rank_num)
    elif args.save_name.startswith('genecis_change_object'):
        dataset = GenecisCOCORerankDataset(ret_query_data_path, ret_cand_data_path, annotation_path=annotation_path, image_path_prefix=image_path_prefix, rank_num=rank_num)
    elif args.save_name.startswith('genecis_focus_object'):
        dataset = GenecisCOCORerankDataset(ret_query_data_path, ret_cand_data_path, annotation_path=annotation_path, image_path_prefix=image_path_prefix, rank_num=rank_num)
    elif args.save_name.startswith('genecis_change_attribute'):
        dataset = GenecisVAWRerankDataset(ret_query_data_path, ret_cand_data_path, annotation_path=annotation_path, image_path_prefix=image_path_prefix, rank_num=rank_num)
    elif args.save_name.startswith('genecis_focus_attribute'):
        dataset = GenecisVAWRerankDataset(ret_query_data_path, ret_cand_data_path, annotation_path=annotation_path, image_path_prefix=image_path_prefix, rank_num=rank_num)
    elif args.save_name.startswith('ccneg'):
        dataset = CCNegRerankDataset(annotation_path)
    elif args.save_name.startswith('sugar_crepe'):
        dataset = SugarCrepeRerankDataset(annotation_path, args.data_type, image_path_prefix)
    elif args.save_name.startswith('circo'):
        dataset = CIRCORerankDataset(ret_query_data_path, ret_cand_data_path, annotation_path, image_path_prefix, split='test', rank_num=rank_num)
    elif args.save_name.startswith('vist'):
        dataset = VistRerankDataset(args.data_path, args.image_path_prefix, args.ret_query_data_path, args.ret_cand_data_path, args.rank_num)
    elif args.save_name.startswith('visdial'):
        dataset = VisDialRerankDataset(args.image_path_prefix, args.data_path, args.ret_query_data_path, args.ret_cand_data_path, args.rank_num)
    elif args.save_name.startswith('mrf'):
        dataset = MultiTurnFashionRerankDataset(args.data_path, annotation_path, image_path_prefix, ret_query_data_path, ret_cand_data_path, category='all', rank_num=args.rank_num)
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
        outputs = model.module.generate(**inputs, max_new_tokens=128, output_scores=True, return_dict_in_generate=True, do_sample=False)
        generated_ids = outputs.sequences
        logits = outputs.scores[0] # (batch_size, 151658)
        scores = []
        for idx in range(len(logits)):
            probs = (
                torch.nn.functional.softmax(
                    torch.FloatTensor(
                        [
                            logits[idx][tokenizer("Yes").input_ids[0]],
                            logits[idx][tokenizer("No").input_ids[0]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            scores.append(probs[0])
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # for item in output_text:
        #     assert item == 'Yes' or item == 'No'

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

        if not args.save_name.startswith('ccneg') and not args.save_name.startswith('sugar_crepe'):
            for query_id in dataset.ret_query_data:
                query_ids.append(query_id)
            for i, query_id in enumerate(query_ids):
                if query_id not in queryid2rerank_score:
                    queryid2rerank_score[query_id] = []
                for j in range(rank_num):
                    queryid2rerank_score[query_id].append(float(filter_scores[i * rank_num + j]))
        else:
            queryid2rerank_score = filter_scores.tolist()

        if not args.save_name.startswith('sugar_crepe'):
            with open(f"./zeroshot_rerank_files/{args.save_name}_test_queryid2rerank_score.json", 'w') as f:
                json.dump(queryid2rerank_score, f, indent=2)
        else:
            with open(f"./zeroshot_rerank_files/{args.save_name}_{args.data_type}_test_queryid2rerank_score.json", 'w') as f:
                json.dump(queryid2rerank_score, f, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--ret_query_data_path', type=str)
    parser.add_argument('--ret_cand_data_path', type=str)
    parser.add_argument('--rank_num', type=int, default=10)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--image_path_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_type', type=str, default=None)
    parser.add_argument('--image_data_path', type=str, default=None)
    parser.add_argument('--text_data_path', type=str, default=None)
    parser.add_argument('--annotation_path', type=str, default=None)

    args = parser.parse_args()
    rerank(args)