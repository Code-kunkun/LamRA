from typing import Dict, List
from torch.utils.data import Dataset
import torch 
import os 
current_file_path = os.path.abspath(__file__)
replace_file_path_prefix = current_file_path.replace("dataset/datasets_ccneg.py", "data/ccneg/")


class CCNegDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        type: str, 
    ) -> None:
        super(CCNegDataset, self).__init__()
        data = torch.load(data_path)
        self.split_size = 40000
        self.image_paths, self.annotations = None, None 

        self.annotations = data['annotations']
        self.image_paths = [path.replace("/workspace/datasets/cc3m/", replace_file_path_prefix) for path in data['image_paths']]
        self.image_paths = self.image_paths[-self.split_size:]
        self.annotations = self.annotations[-self.split_size:]
        self.texts = []
        for annos in self.annotations:
            caption = annos['json']['caption']
            negative_prompt = annos['sop_data']['negative-prompt']
            negative_prompt = negative_prompt.replace(",", "")
            self.texts.append(caption)
            self.texts.append(negative_prompt)
        self.type = type 

    def __len__(self) -> int:
        if self.type == 'image':
            return len(self.image_paths)
        else:
            return len(self.texts)

    def construct_messages(self, text=None, image=None):
        if image is not None and text is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{text}\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        if image is None:
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
        elif text is None:
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
            message = self.construct_messages(image=self.image_paths[index])
        else:
            text = self.texts[index]
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

class CCNegRerankDataset(Dataset):

    def __init__(
        self, 
        annotation_path, 
        rank_num: int = 2
    ) -> None:
        super(CCNegRerankDataset, self).__init__()
        data = torch.load(annotation_path)
        self.split_size = 40000
        self.image_paths, self.annotations = None, None 

        self.annotations = data['annotations']
        self.image_paths = [path.replace("/workspace/datasets/cc3m/", replace_file_path_prefix) for path in data['image_paths']]
        self.image_paths = self.image_paths[-self.split_size:]
        self.annotations = self.annotations[-self.split_size:]
        self.texts = []
        for annos in self.annotations:
            caption = annos['json']['caption']
            negative_prompt = annos['sop_data']['negative-prompt']
            negative_prompt = negative_prompt.replace(",", "")
            self.texts.append(caption)
            self.texts.append(negative_prompt)
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.image_paths) * self.rank_num 

    def construct_rerank_messages(self, query_dict, cand_dict, type='pos'):
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
        instruction = "Find an image caption describing the following everyday image."
        text = self.texts[index]
        image = self.image_paths[index // self.rank_num]
        query_dict = {'image': image, 'txt': instruction}
        cand_dict = {'txt': text}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 
