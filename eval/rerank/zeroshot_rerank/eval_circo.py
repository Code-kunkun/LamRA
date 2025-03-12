import json 


task_name = "circo"
raw_scores = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_test_scores.json"))
rerank_scores = json.load(open(f"./zeroshot_rerank_files/{task_name}_top10_all_test_queryid2rerank_score.json"))

query_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_test_query_names.json"))
cand_names = json.load(open(f"./zeroshot_retrieval_eval_results/{task_name}_test_cand_names.json"))

rerank_candidate_names = []
for idx, query_name in enumerate(query_names):
    raw_candidate_names = cand_names[idx][:10]
    raw_score = raw_scores[idx][:10]
    rerank_score = rerank_scores[str(query_name)]
    final_score = [1 * raw_score[index] + 1 * rerank_score[index] for index in range(len(raw_score))]
    sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
    rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
    rerank_candidate_name.extend(cand_names[idx][10:50])
    rerank_candidate_names.append(rerank_candidate_name)
    

res = {}
for query_id, candidate_names in zip(query_names, rerank_candidate_names):
    res[query_id] = candidate_names

with open('./zeroshot_rerank_files/circo_test_rerank_results.json', 'w') as f:
    json.dump(res, f)