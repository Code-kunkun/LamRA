import json
from typing import Dict, List
from torch.utils.data import Dataset
import pickle 


class MSVDDataset(Dataset):

    def __init__(
        self, 
        video_path_prefix, 
        test_video_path, 
        captions_path,
        type: str='video', 
    ) -> None:
        super(MSVDDataset, self).__init__()
        self.videos = []
        self.texts = []
        with open(captions_path, 'rb') as f:
            self.captions = pickle.load(f)
        with open(test_video_path, 'r') as f:
            test_videos = f.readlines()
        self.text2video_gt_index = []
        for index, item in enumerate(test_videos):
            self.videos.append(video_path_prefix + '/' + item.strip() + '.avi')
            video_captions = self.captions[item.strip()]
            for cap in video_captions:
                self.text2video_gt_index.append(index)
                self.texts.append(' '.join(cap))
        self.video_path_prefix = video_path_prefix
        self.type = type 

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
                        {"type": "video", "video": video, 'nframes': 12},
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
                        {"type": "video", "video": video, 'nframes': 12},
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
            message = self.construct_messages(video=self.videos[index])
        else:
            text = self.texts[index]
            text = f"Find an everyday video match with caption. {text}"
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

class MSVDRerankT2VDataset(Dataset):

    def __init__(
        self,
        ret_query_data_path: str,
        ret_cand_data_path: str,
        video_path_prefix, 
        test_video_path, 
        captions_path,
        rank_num: int = 10, 
    ) -> None:
        super(MSVDRerankT2VDataset, self).__init__()
        self.videos = []
        self.texts = []
        with open(captions_path, 'rb') as f:
            self.captions = pickle.load(f)
        with open(test_video_path, 'r') as f:
            test_videos = f.readlines()
        self.text2video_gt_index = []
        for index, item in enumerate(test_videos):
            self.videos.append(video_path_prefix + '/' + item.strip() + '.avi')
            video_captions = self.captions[item.strip()]
            for cap in video_captions:
                self.text2video_gt_index.append(index)
                self.texts.append(' '.join(cap))
        self.video_path_prefix = video_path_prefix
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
            query.append({'type': 'video', 'video': query_dict['video'], 'nframes': 8})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'video' in cand_dict:
            cand.append({'type': 'video', 'video': cand_dict['video'], 'nframes': 8})
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
        cand_dict = {'video': self.videos[cand_id]}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 
