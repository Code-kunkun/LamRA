import json 
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--data_type', type=str, default=None)

args = parser.parse_args()
task_name = args.task_name 
data_type = args.data_type 

if args.task_name == 'ccneg':
    raw_score = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_scores.json"))
    rerank_score = json.load(open(f"./zeroshot_rerank_files/{task_name}_top2_all_test_queryid2rerank_score.json"))
    beat = 0
    for i in tqdm(range(len(rerank_score) // 2)):
        score1 = 1 * raw_score[2 * i] + 1.0 * rerank_score[2 * i]
        score2 = 1 * raw_score[2 * i + 1] + 1.0 * rerank_score[2 * i + 1]
        if score1 > score2:
            beat += 1
    print(beat / (len(rerank_score) // 2))
else:
    raw_score = json.load(open(f"./zeroshot_retrieval_eval_results/sugar_crepe_{data_type}.json"))
    rerank_score = json.load(open(f"./zeroshot_rerank_files/sugar_crepe_top2_all_{data_type}_test_queryid2rerank_score.json"))
    beat = 0
    for i in tqdm(range(len(rerank_score) // 2)):
        score1 = 1 * raw_score[2 * i] + 0.1 * rerank_score[2 * i]
        score2 = 1 * raw_score[2 * i + 1] + 0.1 * rerank_score[2 * i + 1]
        if score1 > score2:
            beat += 1
    print(beat / (len(rerank_score) // 2))