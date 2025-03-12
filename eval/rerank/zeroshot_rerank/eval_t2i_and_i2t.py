import json 
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)

args = parser.parse_args()
task_name = args.task_name 

raw_scores = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_scores.json"))
rerank_scores = json.load(open(f"./zeroshot_rerank_files/{task_name}_top10_all_test_queryid2rerank_score.json"))

query_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_query_names.json"))
cand_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_cand_names.json"))


rerank_candidate_names = []
for idx, query_name in enumerate(query_names):
    raw_candidate_names = cand_names[idx][:10]
    raw_score = raw_scores[idx][0][:10]
    rerank_score = rerank_scores[str(query_name)]
    final_score = [1 * raw_score[index] + 1 * rerank_score[index] for index in range(len(raw_score))]
    sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
    rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
    rerank_candidate_names.append(rerank_candidate_name)

k_lists = [1, 5, 10]
res = {}

for k in k_lists:
    res[f'recall_{k}'] = []

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

for ind, query_name in enumerate(tqdm(query_names)):
    if args.task_name.startswith('sharegpt') or args.task_name.startswith('urban1k'):
        relevant_docs = [query_name]
    elif args.task_name.startswith('flickr_t2i'):
        relevant_docs = [query_name // 5]
    elif args.task_name.startswith('flickr_i2t'):
        relevant_docs = [query_name * 5, query_name * 5 + 1, query_name * 5 + 2, query_name * 5 + 3, query_name * 5 + 4]
    retrieved_indices_for_qid = rerank_candidate_names[ind]
    for k in k_lists:
        recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
        res[f'recall_{k}'].append(recall_at_k)

for k in k_lists:
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")