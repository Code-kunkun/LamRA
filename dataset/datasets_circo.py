import json
from typing import Dict, List
from torch.utils.data import Dataset


class CIRCODataset(Dataset):

    def __init__(
        self, 
        annotation_path_prefix,
        image_path_prefix,
        split='test',
        type: str="query"
    ) -> None:

        super(CIRCODataset, self).__init__()
        img_info_path = f"{annotation_path_prefix}/image_info_unlabeled2017.json"
        with open(img_info_path) as f:
            imgs_info = json.load(f)
        
        self.img_paths = [f"{image_path_prefix}/{img_info['file_name']}" for img_info in imgs_info['images']]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}
        annotation_file = f"{annotation_path_prefix}/{split}.json"
        with open(annotation_file) as f:
            self.annotations = json.load(f)        
        self.max_num_gts = 23
        self.type = type 


    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.annotations)
        elif self.type == 'image':
            return len(self.img_paths)

    def construct_messages(self, text=None, image=None):
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
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{text}"},
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

        return message

    def get_instance(self, index):
        if self.type == 'query':
            instruction = "I'm looking for a similar everyday image with the described changes."
            query_id = str(self.annotations[index]['id'])
            relative_caption = self.annotations[index]['relative_caption']
            relative_caption = f"{instruction} {relative_caption}"
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            query_message = self.construct_messages(text=relative_caption, image=reference_img_path)
            return query_message
        elif self.type == 'image':
            image = self.img_paths[index]
            candidate_message = self.construct_messages(image=image)
            return candidate_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class CIRCORerankDataset(Dataset):

    def __init__(
        self, 
        ret_query_data_path: str, 
        ret_cand_data_path: str,
        annotation_path_prefix: str,
        image_path_prefix: str,
        split='val',
        type: str="query",
        rank_num: int = 50,
    ) -> None:

        super(CIRCORerankDataset, self).__init__()
        img_info_path = f"{annotation_path_prefix}/image_info_unlabeled2017.json"
        with open(img_info_path) as f:
            imgs_info = json.load(f)
        
        self.img_paths = [f"{image_path_prefix}/{img_info['file_name']}" for img_info in imgs_info['images']]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}
        annotation_file = f"{annotation_path_prefix}/{split}.json"
        with open(annotation_file) as f:
            self.annotations = json.load(f)        
        self.max_num_gts = 23
        self.type = type 
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 


    def __len__(self) -> int:
        return len(self.annotations) * self.rank_num 

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
        instruction = "I'm looking for a similar everyday image with the described changes"
        relative_caption = self.annotations[index // self.rank_num]['relative_caption']
        relative_caption = f"{instruction} {relative_caption}"
        reference_img_id = str(self.annotations[index // self.rank_num]['reference_img_id'])
        reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
        query_dict = {'image': reference_img_path, 'txt': relative_caption}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_img_path = self.img_paths[self.img_ids_indexes_map[cand_id]]
        cand_dict = {'image': cand_img_path}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 
