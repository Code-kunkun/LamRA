import collections
import os
import json
from typing import Dict, List
from torch.utils.data import Dataset
from tqdm import tqdm 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VistDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        image_path_prefix: str, 
        type: str, 
    ) -> None:
        super(VistDataset, self).__init__()
        self.image_path_prefix = image_path_prefix
        vist_data_raw = json.load(open(data_path))
        vist_data = {
            'annotations': collections.defaultdict(list)
        }
        used_image_ids = []
        
        for ann in tqdm(vist_data_raw['annotations']):
            assert len(ann) == 1
            ann = ann[0]
            story_id = ann['story_id']
            vist_data['annotations'][story_id].append({
                'caption': ann['text'],
                'image_id': ann['photo_flickr_id'],
                'sequence_index': ann['worker_arranged_photo_order']
            })
            used_image_ids.append(ann['photo_flickr_id'])
        self.used_image_ids = sorted(list(set(used_image_ids)))

        self.images = []
        self.imageid2path = {}
        
        image_files = set(os.listdir(self.image_path_prefix))
        for image_id in self.used_image_ids:
            image_suffix1 = f"{image_id}.jpg"
            image_suffix2 = f"{image_id}.png"
            image_suffix3 = f"{image_id}.gif"
            if image_suffix1 in image_files:
                image_path = f"{self.image_path_prefix}/{image_suffix1}"
                self.imageid2path[image_id] = image_suffix1
            elif image_suffix2 in image_files:
                image_path = f"{self.image_path_prefix}/{image_suffix2}"
                self.imageid2path[image_id] = image_suffix2
            else:
                image_path = f"{self.image_path_prefix}/{image_suffix3}"
                self.imageid2path[image_id] = image_suffix3
            self.images.append(image_path)

        assert len(self.used_image_ids) == len(self.images)

        self.story_data = vist_data['annotations']
        self.type = type 

    def __len__(self) -> int:
        if self.type == 'image':
            return len(self.images)
        else:
            return len(self.story_data)

    def construct_messages(self, text=None, image=None):
        if type(image) == list:
            message = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "I will provide you with a series of images and captions from a story, and I need you to retrieve the image corresponding to the last caption."}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
            for cap, single_image in zip(text[:-1], image[:-1]):
                message[0]['content'].append({'type': 'text', 'text': cap})
                message[0]['content'].append({'type': 'image', 'image': single_image})
            message[0]['content'].append({'type': 'text', 'text': text[-1]})
            message[0]['content'].append({"type": "text", "text": f"\nSummarize above images and sentences in one word: "})
        elif image is not None and text is not None:
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
            return message, self.used_image_ids[index]
        else:
            story_data = list(self.story_data.values())[index]
            image_ids = [s['image_id'] for s in story_data[:-1]]
            captions = [s['caption'] for s in story_data]
            assert len(image_ids) == len(captions) - 1
            images = [f"{self.image_path_prefix}/{self.imageid2path[item]}" for item in image_ids]
            message = self.construct_messages(text=captions, image=images)
            return message, index 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i) 

class VistRerankDataset(Dataset):

    def __init__(
        self, 
        annotation_path: str, 
        image_path_prefix: str,  
        ret_query_data_path: str, 
        ret_cand_data_path: str,
        rank_num: int = 10
    ) -> None:
        super(VistRerankDataset, self).__init__()
        self.image_path_prefix = image_path_prefix
        vist_data_raw = json.load(open(annotation_path))
        vist_data = {
            'annotations': collections.defaultdict(list)
        }
        used_image_ids = []
        
        for ann in tqdm(vist_data_raw['annotations']):
            assert len(ann) == 1
            ann = ann[0]
            story_id = ann['story_id']
            vist_data['annotations'][story_id].append({
                'caption': ann['text'],
                'image_id': ann['photo_flickr_id'],
                'sequence_index': ann['worker_arranged_photo_order']
            })
            used_image_ids.append(ann['photo_flickr_id'])
        self.used_image_ids = sorted(list(set(used_image_ids)))

        self.images = []
        self.imageid2path = {}
        
        image_files = set(os.listdir(self.image_path_prefix))
        for image_id in self.used_image_ids:
            image_suffix1 = f"{image_id}.jpg"
            image_suffix2 = f"{image_id}.png"
            image_suffix3 = f"{image_id}.gif"
            if image_suffix1 in image_files:
                image_path = f"{self.image_path_prefix}/{image_suffix1}"
                self.imageid2path[image_id] = image_suffix1
            elif image_suffix2 in image_files:
                image_path = f"{self.image_path_prefix}/{image_suffix2}"
                self.imageid2path[image_id] = image_suffix2
            else:
                image_path = f"{self.image_path_prefix}/{image_suffix3}"
                self.imageid2path[image_id] = image_suffix3
            self.images.append(image_path)

        assert len(self.used_image_ids) == len(self.images)

        self.story_data = vist_data['annotations']

        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num

    def __len__(self):
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
        for cap, single_image in zip(query_dict['txt'][:-1], query_dict['image']):
            query.append({'type': 'text', 'text': cap})
            query.append({'type': 'image', 'image': single_image})
        query.append({'type': 'text', 'text': query_dict['txt'][-1]})

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
        instruction = "I will provide you with a series of images and captions from a story, and I need you to retrieve the image corresponding to the last caption."
        story_data = list(self.story_data.values())[index // self.rank_num]
        image_ids = [s['image_id'] for s in story_data[:-1]]
        captions = [s['caption'] for s in story_data]
        assert len(image_ids) == len(captions) - 1 
        images = [f"{self.image_path_prefix}/{self.imageid2path[item]}" for item in image_ids]
        query_dict = {'image': images, 'txt': captions}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'image': f"{self.image_path_prefix}/{self.imageid2path[cand_id]}"}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict, instruction)
        return rerank_message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 
