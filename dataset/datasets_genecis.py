import json
from typing import Dict, List
from torch.utils.data import Dataset

class GenecisCOCODataset(Dataset):

    def __init__(
        self, 
        annotation_path,
        image_path_prefix,
        type: str="query"
    ) -> None:

        super(GenecisCOCODataset, self).__init__()
        self.type = type 
        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for item in self.val_samples:
            self.gallery_ids.add(str(item['target']['val_image_id']))
            gallery = item['gallery']
            for x in gallery:
                self.gallery_ids.add(str(x['val_image_id']))

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.val_samples)
        elif self.type == 'image':
            return len(self.gallery_ids)

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
            sample = self.val_samples[index]
            reference_name = str(sample['reference']['val_image_id'])
            reference_img_path = f"{self.image_path_prefix}/{reference_name.zfill(12)}.jpg"
            relative_caption = sample['condition']
            relative_caption = f"{instruction} {relative_caption}"
            query_message = self.construct_messages(text=relative_caption, image=reference_img_path)
            return query_message
        elif self.type == 'image':
            image_id = self.gallery_ids[index]
            image = f"{self.image_path_prefix}/{image_id.zfill(12)}.jpg"
            candidate_message = self.construct_messages(image=image)
            return candidate_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 



class GenecisVAWDataset(Dataset):

    def __init__(
        self, 
        annotation_path,
        image_path_prefix,
        type: str="query",
    ) -> None:

        super(GenecisVAWDataset, self).__init__()
        self.type = type 

        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for index, item in enumerate(self.val_samples):
            self.gallery_ids.add(f"{str(item['target']['image_id'])}_{index}_1.jpg")
            gallery = item['gallery']
            for i, x in enumerate(gallery):
                self.gallery_ids.add(f"{str(x['image_id'])}_{index}_{2 + i}.jpg")

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.val_samples)
        elif self.type == 'image':
            return len(self.gallery_ids)

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
            if 'change_attribute' in self.image_path_prefix:
                instruction = "I'm looking for a similar everyday image with the described changes."
            elif 'focus_attribute' in self.image_path_prefix:
                instruction = "I'm looking for an image with the same attributes as described."
            sample = self.val_samples[index]
            reference_name = str(sample['reference']['image_id'])
            reference_img_path = f"{self.image_path_prefix}/{reference_name}_{index}_0.jpg"
            relative_caption = sample['condition']
            relative_caption = f"{instruction} {relative_caption}"
            query_message = self.construct_messages(text=relative_caption, image=reference_img_path)
            return query_message
        elif self.type == 'image':
            image_id = self.gallery_ids[index]
            image = f"{self.image_path_prefix}/{image_id}"
            candidate_message = self.construct_messages(image=image)
            return candidate_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 



class GenecisCOCORerankDataset(Dataset):

    def __init__(
        self, 
        ret_query_data_path: str,
        ret_cand_data_path: str,
        annotation_path: str,
        image_path_prefix: str,
        split='val',
        type: str="query",
        rank_num: int=10
    ) -> None:

        super(GenecisCOCORerankDataset, self).__init__()
        self.type = type 
        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for item in self.val_samples:
            self.gallery_ids.add(str(item['target']['val_image_id']))
            gallery = item['gallery']
            for x in gallery:
                self.gallery_ids.add(str(x['val_image_id']))

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num

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
        instruction = "I'm looking for a similar everyday image with the described changes."
        sample = self.val_samples[index // self.rank_num]
        reference_name = str(sample['reference']['val_image_id'])
        reference_img_path = f"{self.image_path_prefix}/{reference_name.zfill(12)}.jpg"
        relative_caption = sample['condition']
        relative_caption = f"{instruction} {relative_caption}"
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        target_name = str(sample['target']['val_image_id'])
        gallery_names = [str(item['val_image_id']) for item in sample['gallery']]
        target_and_gallery_names = [target_name]
        target_and_gallery_names.extend(gallery_names)
        cand_name = target_and_gallery_names[cand_id]
        cand_img_path = f"{self.image_path_prefix}/{cand_name.zfill(12)}.jpg"
        query_dict = {'image': reference_img_path, 'txt': relative_caption}
        cand_dict = {'image': cand_img_path}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class GenecisVAWRerankDataset(Dataset):

    def __init__(
        self, 
        ret_query_data_path: str,
        ret_cand_data_path: str, 
        annotation_path: str,
        image_path_prefix: str,
        type: str="query",
        rank_num: int=10
    ) -> None:

        super(GenecisVAWRerankDataset, self).__init__()
        self.type = type 

        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for index, item in enumerate(self.val_samples):
            self.gallery_ids.add(f"{str(item['target']['image_id'])}_{index}_1.jpg")
            gallery = item['gallery']
            for i, x in enumerate(gallery):
                self.gallery_ids.add(f"{str(x['image_id'])}_{index}_{2 + i}.jpg")

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num 

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
        if 'change_attribute' in self.image_path_prefix:
            instruction = "I'm looking for a similar everyday image with the described changes."
        elif 'focus_attribute' in self.image_path_prefix:
            instruction = "I'm looking for an image with the same attributes as described."
        sample = self.val_samples[index // self.rank_num]
        reference_name = str(sample['reference']['image_id'])
        reference_img_path = f"{self.image_path_prefix}/{reference_name}_{index // self.rank_num}_0.jpg"
        relative_caption = sample['condition']
        relative_caption = f"{instruction} {relative_caption}"
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        target_name = f"{str(sample['target']['image_id'])}_{index // self.rank_num}_1.jpg"
        gallery_names = [f"{str(item['image_id'])}_{index // self.rank_num}_{2 + idx}.jpg" for idx, item in enumerate(sample['gallery'])]
        target_and_gallery_names = [target_name]
        target_and_gallery_names.extend(gallery_names) 
        cand_name = target_and_gallery_names[cand_id]
        cand_img_path = f"{self.image_path_prefix}/{cand_name}"
        query_dict = {'image': reference_img_path, 'txt': relative_caption}
        cand_dict = {'image': cand_img_path}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 