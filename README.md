
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=Code-kunkun&project=LamRA&lang=as">অসমীয়া</
      </div>
    </div>
  </details>
</div>

# LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant

This repository is the official implementation of LamRA.

[🏡 Project Page](https://code-kunkun.github.io/LamRA/) |  [📄 Paper](https://arxiv.org/pdf/2412.01720) | [🤗 LamRA-Ret-Pretrained](https://huggingface.co/code-kunkun/LamRA-Ret-Pretrained) | [🤗 LamRA-Ret](https://huggingface.co/code-kunkun/LamRA-Ret) | [🤗 LamRA-Rank](https://huggingface.co/code-kunkun/LamRA-Rank) | [🤗 Dataset](https://huggingface.co/datasets/code-kunkun/LamRA_Eval)

## Installation

```bash 
conda create -n lamra python=3.10 -y
conda activate lamra 

pip install --upgrade pip  # enable PEP 660 support 
pip install -r requirements.txt

pip install ninja
pip install flash-attn --no-build-isolation
```

## New Version
We have updated the version of Qwen2.5-VL in the `qwen2.5vl` branch.

## Quickstart
Please refer to the `demo.py`

## Data Preparation 

Download Qwen2-VL-7B and place it in `./checkpoints/hf_models/Qwen2-VL-7B-Instruct`

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
@inproceedings{liu2025lamra,
  title={Lamra: Large multimodal model as your advanced retrieval assistant},
  author={Liu, Yikun and Zhang, Yajie and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Yao, Jiangchao and Wang, Yanfeng and Xie, Weidi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4015--4025},
  year={2025}
}
```