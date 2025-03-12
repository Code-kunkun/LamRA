import json
from torch.utils.data import Dataset

class SugarCrepeDataset(Dataset):

    def __init__(
        self,
        annotation_path_prefix,
        data_type,
        image_path_prefix,
        type,
    ):
        annotation_path = f"{annotation_path_prefix}/{data_type}.json" 
        self.annotations = json.load(open(annotation_path))
        self.type = type 
        self.image_paths = []
        self.texts = []
        for annos_key in self.annotations:
            annos = self.annotations[annos_key]
            image_path = f"{image_path_prefix}/{annos['filename']}"
            self.image_paths.append(image_path)
            self.texts.append(annos['caption'])
            self.texts.append(annos['negative_caption'])
        self.type = type 

    def __len__(self):
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

    def __getitem__(self, i):
        return self.get_instance(i), i  


class SugarCrepeRerankDataset(Dataset):

    def __init__(
        self,
        annotation_path_prefix,
        data_type,
        image_path_prefix,
        rank_num: int = 2
    ):
        annotation_path = f"{annotation_path_prefix}/{data_type}.json" 
        self.annotations = json.load(open(annotation_path))
        self.image_paths = []
        self.texts = []
        for annos_key in self.annotations:
            annos = self.annotations[annos_key]
            image_path = f"{image_path_prefix}/{annos['filename']}"
            self.image_paths.append(image_path)
            self.texts.append(annos['caption'])
            self.texts.append(annos['negative_caption'])
        self.rank_num = rank_num 

    def __len__(self):
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

    def __getitem__(self, i):
        return self.get_instance(i), i  