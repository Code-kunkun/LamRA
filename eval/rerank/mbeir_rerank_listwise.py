import json 
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)

args = parser.parse_args()
task_name = args.task_name 

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

qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
qrel, _ = load_qrel(qrels_path)

query_names = json.load(open(f"./LamRA_Ret_eval_results/mbeir_{task_name}_test_qwen2-vl-7b_LamRA-Ret_query_names.json"))
cand_names = json.load(open(f"./LamRA_Ret_eval_results/mbeir_{task_name}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json"))

rerank_path_prefix = "./mbeir_rerank_files"
rerank_data = json.load(open(f"{rerank_path_prefix}/{task_name}_top5_all_test_queryid2rerank_outputs_listwise.json"))

rerank_candidate_names = []
for idx, query_name in enumerate(query_names):
    raw_candidate_names = cand_names[idx][:5]
    top1_name = rerank_data[query_name]
    if top1_name == 'Ans: (1)':
        rerank_candidate_names.append([raw_candidate_names[0]])
    elif top1_name == 'Ans: (2)':
        rerank_candidate_names.append([raw_candidate_names[1]])
    elif top1_name == 'Ans: (3)':
        rerank_candidate_names.append([raw_candidate_names[2]])
    elif top1_name == 'Ans: (4)':
        rerank_candidate_names.append([raw_candidate_names[3]])
    elif top1_name == 'Ans: (5)':
        rerank_candidate_names.append([raw_candidate_names[4]])  

# rerank_candidate_names = cand_names 

k_lists = [1]
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
    relevant_docs = qrel[query_name]
    retrieved_indices_for_qid = rerank_candidate_names[ind]
    for k in k_lists:
        recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
        res[f'recall_{k}'].append(recall_at_k)

for k in k_lists:
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")  