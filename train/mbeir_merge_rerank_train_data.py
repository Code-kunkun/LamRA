import os 
import json 
from tqdm import tqdm 

path_prefix = './rerank_training_data'
files = os.listdir(path_prefix)

rerank_all_data = {}

for file in tqdm(files):
    data = json.load(open(path_prefix + '/' + file))
    for item in data:
        rerank_all_data[item] = data[item]

with open(path_prefix + '/' + 'rerank_data_all.json', 'w') as f:
    json.dump(rerank_all_data, f)