import json
from typing import Dict, List
from torch.utils.data import Dataset
import pandas as pd 


class MSRVTTDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        video_data_path: str, 
        type: str, 
    ) -> None:
        super(MSRVTTDataset, self).__init__()
        self.images = []
        self.texts = []
        self.data = pd.read_csv(data_path)
        self.video_data_path = video_data_path
        self.type = type 

        self.videos = self.data['video_id'].tolist()
        self.texts = self.data['sentence'].tolist()

    def __len__(self) -> int:
        if self.type == 'video':
            return len(self.videos)
        else:
            return len(self.texts)

    def construct_messages(self, text=None, video=None):
        if video is not None and text is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": self.video_data_path + '/' + video + '.mp4'},
                        {"type": "text", "text": f"{text}\nSummarize above video and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif video is None:
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
                        {"type": "video", "video": self.video_data_path + '/' + video + '.mp4'},
                        {"type": "text", "text": f"\nSummarize above video in one word: "}
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
        if self.type == 'video':
            # text = "Find a caption describe the video."
            message = self.construct_messages(video=self.videos[index])
            # message = self.construct_messages(video=self.videos[index], text=text)
        else:
            text = self.texts[index]
            text = f"Find an everyday video match with caption. {text}"
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class MSRVTTRerankT2VDataset(Dataset):

    def __init__(
        self, 
        data_path: str,
        video_data_path: str,
        ret_query_data_path: str,
        ret_cand_data_path: str,
        rank_num: int = 10,
    ) -> None:
        super(MSRVTTRerankT2VDataset, self).__init__()
        self.images = []
        self.texts = []
        self.data = pd.read_csv(data_path)
        self.video_data_path = video_data_path
        self.type = type 

        self.videos = self.data['video_id'].tolist()
        self.texts = self.data['sentence'].tolist()
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.texts) * self.rank_num 

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

        if 'video' in query_dict:
            query.append({'type': 'video', 'video': query_dict['video']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'video' in cand_dict:
            cand.append({'type': 'video', 'video': cand_dict['video']})
        if 'txt' in cand_dict:
            cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def get_instance(self, index):
        instruction = "Find an everyday video match with caption."
        text = self.texts[index // self.rank_num]
        query_dict = {'txt': f"{instruction} {text}"}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'video': self.video_data_path + '/' + self.videos[cand_id] + '.mp4'}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

