import json 
import collections 
from tqdm import tqdm 

task_name = "vist"
raw_scores = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_scores.json"))
rerank_scores = json.load(open(f"./zeroshot_rerank_files/{task_name}_top10_all_test_queryid2rerank_score.json"))

query_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_query_names.json"))
cand_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_cand_names.json"))
data_path = "./data/vist/sis/val.story-in-sequence.json"
vist_data_raw = json.load(open(data_path))
vist_data = {
    'annotations': collections.defaultdict(list)
}
for ann in tqdm(vist_data_raw['annotations']):
    assert len(ann) == 1
    ann = ann[0]
    story_id = ann['story_id']
    vist_data['annotations'][story_id].append({
        'caption': ann['text'],
        'image_id': ann['photo_flickr_id'],
        'sequence_index': ann['worker_arranged_photo_order']
    })
story_data = vist_data['annotations']
story_data = list(story_data.values())

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
    relevant_docs = [story_data[ind][-1]['image_id']]
    retrieved_indices_for_qid = rerank_candidate_names[ind]
    for k in k_lists:
        recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
        res[f'recall_{k}'].append(recall_at_k)

for k in k_lists:
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")