import os
import json
from typing import Dict, List
from torch.utils.data import Dataset
from tqdm import tqdm 


class VisDialDataset(Dataset):

    def __init__(
        self, 
        image_path_prefix: str, 
        data_path: str, 
        type: str, 
    ) -> None:
        super(VisDialDataset, self).__init__()
        self.image_path_prefix = image_path_prefix
        visdial_data = json.load(open(data_path))
        self.texts = []
        self.images = []
        
        questions = visdial_data['data']['questions']
        answers = visdial_data['data']['answers']
        dialogs = visdial_data['data']['dialogs']
        for example_idx in tqdm(range(len(dialogs))):
            dialog = dialogs[example_idx]
            image_id = str(dialog['image_id']).rjust(12, '0')
            contexts = []
            image_path = os.path.join(self.image_path_prefix, f'VisualDialog_val2018_{image_id}.jpg') 
            self.images.append(image_path)
            for i in range(len(dialog['dialog'])):
                contexts.append('Q: ' + questions[dialog['dialog'][i]['question']] + '?')
                contexts.append('A: ' + answers[dialog['dialog'][i]['answer']] + '.')
            
            full_context_sent = " ".join(contexts)
            self.texts.append(full_context_sent)

        self.type = type 

    def __len__(self) -> int:
        if self.type == 'image':
            return len(self.images)
        else:
            return len(self.texts)

    def construct_messages(self, text=None, image=None):
        if image is not None and text is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": {text}},
                        {"type": "text", "text": f"\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif image is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\nSummarize above image in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        return message

    def get_instance(self, index):
        if self.type == 'image':
            message = self.construct_messages(image=self.images[index])
        else:
            text = self.texts[index]
            instruction = "Find me an everyday image that matches the given dialog."
            text = f"{instruction} {text}"
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

class VisDialRerankDataset(Dataset):

    def __init__(
        self, 
        image_path_prefix: str, 
        data_path: str, 
        ret_query_data_path: str,
        ret_cand_data_path: str,
        rank_num: int = 10
    ) -> None:
        super(VisDialRerankDataset, self).__init__()
        self.image_path_prefix = image_path_prefix
        visdial_data = json.load(open(data_path))
        self.texts = []
        self.images = []
        
        questions = visdial_data['data']['questions']
        answers = visdial_data['data']['answers']
        dialogs = visdial_data['data']['dialogs']
        for example_idx in tqdm(range(len(dialogs))):
            dialog = dialogs[example_idx]
            image_id = str(dialog['image_id']).rjust(12, '0')
            contexts = []
            image_path = os.path.join(self.image_path_prefix, f'VisualDialog_val2018_{image_id}.jpg') 
            self.images.append(image_path)
            for i in range(len(dialog['dialog'])):
                contexts.append('Q: ' + questions[dialog['dialog'][i]['question']] + '?')
                contexts.append('A: ' + answers[dialog['dialog'][i]['answer']] + '.')
            
            full_context_sent = " ".join(contexts)
            self.texts.append(full_context_sent)
            
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num

    def construct_rerank_messages(self, query_dict, cand_dict):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, responed with 'No'."}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidate:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'image' in cand_dict:
            cand.append({'type': 'image', 'image': cand_dict['image']})
        if 'txt' in cand_dict:
            cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def get_instance(self, index):
        instruction = 'Find the image that matches the dialogue'
        text = self.texts[index // self.rank_num]
        text = f"{instruction} {text}"
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        image_path = self.images[cand_id]
        query_dict = {'txt': text}
        cand_dict = {'image': image_path}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 
