# get pointwise reranking results
python eval/rerank/mbeir_rerank.py --task_name visualnews_task0
python eval/rerank/mbeir_rerank.py --task_name mscoco_task0
python eval/rerank/mbeir_rerank.py --task_name fashion200k_task0
python eval/rerank/mbeir_rerank.py --task_name webqa_task1
python eval/rerank/mbeir_rerank.py --task_name edis_task2
python eval/rerank/mbeir_rerank.py --task_name webqa_task2
python eval/rerank/mbeir_rerank.py --task_name visualnews_task3
python eval/rerank/mbeir_rerank.py --task_name mscoco_task3
python eval/rerank/mbeir_rerank.py --task_name fashion200k_task3
python eval/rerank/mbeir_rerank.py --task_name nights_task4
python eval/rerank/mbeir_rerank.py --task_name oven_task6
python eval/rerank/mbeir_rerank.py --task_name infoseek_task6
python eval/rerank/mbeir_rerank.py --task_name fashioniq_task7
python eval/rerank/mbeir_rerank.py --task_name cirr_task7
python eval/rerank/mbeir_rerank.py --task_name oven_task8
python eval/rerank/mbeir_rerank.py --task_name infoseek_task8

# get listwise reranking results
python eval/rerank/mbeir_rerank_listwise.py --task_name visualnews_task0
python eval/rerank/mbeir_rerank_listwise.py --task_name mscoco_task0
python eval/rerank/mbeir_rerank_listwise.py --task_name fashion200k_task0
python eval/rerank/mbeir_rerank_listwise.py --task_name webqa_task1
python eval/rerank/mbeir_rerank_listwise.py --task_name edis_task2
python eval/rerank/mbeir_rerank_listwise.py --task_name webqa_task2
python eval/rerank/mbeir_rerank_listwise.py --task_name visualnews_task3
python eval/rerank/mbeir_rerank_listwise.py --task_name mscoco_task3
python eval/rerank/mbeir_rerank_listwise.py --task_name fashion200k_task3
python eval/rerank/mbeir_rerank_listwise.py --task_name nights_task4
python eval/rerank/mbeir_rerank_listwise.py --task_name oven_task6
python eval/rerank/mbeir_rerank_listwise.py --task_name infoseek_task6
python eval/rerank/mbeir_rerank_listwise.py --task_name fashioniq_task7
python eval/rerank/mbeir_rerank_listwise.py --task_name cirr_task7
python eval/rerank/mbeir_rerank_listwise.py --task_name oven_task8
python eval/rerank/mbeir_rerank_listwise.py --task_name infoseek_task8