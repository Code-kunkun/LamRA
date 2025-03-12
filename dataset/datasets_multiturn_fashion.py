import json
from typing import Dict, List
from torch.utils.data import Dataset


class MultiTurnFashionDataset(Dataset):

    def __init__(
        self, 
        query_data_path: str, 
        cand_data_path: str, 
        image_path_prefix: str,
        type: str, 
        category: str = 'all'
    ) -> None:
        super(MultiTurnFashionDataset, self).__init__()
        self.category = category 
        self.image_path_prefix = image_path_prefix
        query_data_path = query_data_path
        cand_data_path = cand_data_path
        self.query_data = json.load(open(query_data_path))
        self.cand_data = json.load(open(cand_data_path))
        self.type = type 

    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.query_data)
        else:
            return len(self.cand_data)

    def construct_messages(self, text=None, image=None):
        if image is not None and text is not None:
            instruction = "Find me a similar fashion image based on the following multi-round modifications."
            message = [
                {
                    "role": "user",
                    "content": [
                        {'type': 'text', 'text': instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]

            for reference_image, reference_text in zip(image, text):
                message[0]['content'].append({"type": "image", "image": reference_image})
                message[0]['content'].append({"type": "text", "text": reference_text})
            message[0]['content'].append({"type": "text", "text": f"\nSummarize above image and sentence in one word: "})

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
        if self.type == 'query':
            gt_image_name = self.query_data[index]['target'][1]
            reference = self.query_data[index]['reference']
            reference_images = [f"{self.image_path_prefix}/{item[2]}.jpg" for item in reference]
            
            reference_texts = []
            for item in reference:
                texts = item[1]
                texts = [text for text in texts if text != ""]
                reference_texts.append(" and ".join(texts))

            message = self.construct_messages(image=reference_images, text=reference_texts)
            # return message, gt_image_name 
            return message, index 

        elif self.type == 'cand':
            image_name = self.cand_data[index]
            image_path = f"{self.image_path_prefix}/{image_name}.jpg"
            message = self.construct_messages(image=image_path)
            return message, image_name 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i)


class MultiTurnFashionRerankDataset(Dataset):

    def __init__(
        self, 
        query_data_path: str, 
        cand_data_path: str, 
        image_path_prefix: str,
        ret_query_data_path: str,
        ret_cand_data_path: str,
        category: str = 'all',
        rank_num: int = 10
    ) -> None:
        super(MultiTurnFashionRerankDataset, self).__init__()
        self.category = category 
        self.image_path_prefix = image_path_prefix
        query_data_path = query_data_path
        cand_data_path = cand_data_path
        self.query_data = json.load(open(query_data_path))
        self.cand_data = json.load(open(cand_data_path))
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num

    def construct_rerank_messages(self, query_dict, cand_dict, instruction):
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
        query.append({'type': 'text', 'text': instruction})
        for single_image, cap in zip(query_dict['image'], query_dict['txt']):
            query.append({'type': 'image', 'image': single_image})
            query.append({'type': 'text', 'text': cap})

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
        instruction = "Find me a similar fashion image based on the following multi-round modifications."
        reference = self.query_data[index // self.rank_num]['reference']
        reference_images = [f"{self.image_path_prefix}/{item[2]}.jpg" for item in reference]
        
        reference_texts = []
        for item in reference:
            texts = item[1]
            texts = [text for text in texts if text != ""]
            reference_texts.append(" and ".join(texts))

        query_dict = {'image': reference_images, 'txt': reference_texts}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'image': f"{self.image_path_prefix}/{cand_id}.jpg"}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict, instruction)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i

