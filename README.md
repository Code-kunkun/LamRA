# LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant

This repository is the official implementation of LamRA.

[🏡 Project Page](https://code-kunkun.github.io/LamRA/) |  [📄 Paper](https://arxiv.org/pdf/2412.01720) | [🤗 LamRA-Ret-Qwen2.5VL](https://huggingface.co/code-kunkun/LamRA-Ret-Qwen2.5VL-7b) | [🤗 LamRA-Rank-Qwen2.5VL](https://huggingface.co/code-kunkun/LamRA-Rank-Qwen2.5VL-7b) | [🤗 Dataset](https://huggingface.co/datasets/code-kunkun/LamRA_Eval)


## Installation

```bash 
conda create -n lamra python=3.10 -y
conda activate lamra 

pip install --upgrade pip  # enable PEP 660 support 
pip install -r requirements.txt

pip install ninja
pip install flash-attn --no-build-isolation
```

## Transformers Version
As mentioned in the [issue of Qwen2.5VL](https://github.com/QwenLM/Qwen2.5-VL/issues/706), please install the environment according to the above instruction.

Change the `apply_rotary_pos_emb_flashatt` function in `modeling_qwen2_5_vl.py`.
```python
def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q) # revise here
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k) # revise here
    return q_embed, k_embed
```

## Quickstart
Please refer to the `demo.py`

## Data Preparation 

Download Qwen2.5-VL-7B and place it in `./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct`

For pre-training dataset, please refer to [link](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse)

For multimodal instruction tuning datset, please refer to [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)

For evaluation data related to the LamRA, please refer to [LamRA_Eval](https://huggingface.co/datasets/code-kunkun/LamRA_Eval)

After downloading all of them, organize the data as follows in `./data`
```
├── M-BEIR
├── nli_for_simcse.csv
├── rerank_data_for_training
├── flickr
├── coco
├── sharegpt4v
├── Urban1K
├── circo
├── genecis
├── vist
├── visdial
├── ccneg
├── sugar-crepe
├── MSVD
└── msrvtt
```

## Training & Evaluation for LamRA-Ret

### Pre-training

```bash 
sh scripts/lamra_ret/pretrain.sh
```

```bash 
# Evaluation 
sh scripts/eval/eval_pretrained.sh
```

```bash 
# Merge LoRA for multimodal instruction tuning stage
sh scripts/merge_lora.sh 
```

###  Multimodal instruction tuning

```bash
sh scripts/lamra_ret/finetune.sh
```

```bash 
# Evaluation 
sh scripts/eval/eval_mbeir.sh   # eval under local pool setting

sh scripts/eval/eval_mbeir_global.sh   # eval under global pool setting
```

## Training & Evaluation for LamRA-Rank

You can use the [data](https://huggingface.co/datasets/code-kunkun/LamRA_Eval/tree/main/rerank_data_for_training) we provide or run the following command to get the data for reranking training.

```bash
# Collecting data for reranking training
sh scripts/lamra_rank/get_train_data.sh

sh scripts/lamra_rank/merge_train_data.sh
```

```bash
# training for reranking
sh scripts/lamra_rank/train_rerank.sh
```

```bash 
# pointwise reranking
sh scripts/eval/eval_rerank_mbeir_pointwise.sh

# listwise reranking
sh scripts/eval/eval_rerank_mbeir_listwise.sh
```

```bash
# Get the reranking results on M-BEIR
sh scirpts/eval/get_rerank_results_mbeir.sh
```

## Evaluation on other benchmarks

```bash
# evaluation results on zeroshot datasets
sh scirpts/eval/eval_zeroshot.sh

# reranking the results on zeroshot datasets
sh scripts/eval/eval_rerank_zeroshot.sh

# get the final results
sh scripts/eval/get_rerank_results_zeroshot.sh
```


## 🫡 Acknowledgements

Many thanks to the code bases from [lmms-finetune](https://github.com/zjysteven/lmms-finetune) and [E5-V](https://github.com/kongds/E5-V).


## Citation
If you use this code for your research or project, please cite:
```latex
@article{liu2024lamra,
  title={LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant},
  author={Yikun Liu and Pingan Chen and Jiayin Cai and Xiaolong Jiang and Yao Hu and Jiangchao Yao and Yanfeng Wang and Weidi Xie},
  journal={arXiv preprint arXiv:2412.01720},
  year={2024}
}
```
