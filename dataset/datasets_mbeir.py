import os
import json
from torch.utils.data import Dataset
import random 


DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        tokenizer = None 
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.tokenizer = tokenizer 
        self.image_path_prefix = image_path_prefix 

    def __len__(self) -> int:
        return len(self.query_data)

    
    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'txt' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
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
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(f"{query_txt}")
        pos_img_path = pos_cand.get("img_path", None)

        # truncation processing is applied to prevent memory overflow.
        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])
        # query_txt_without_prompt = self.tokenizer(query_txt_without_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # query_txt_without_prompt = self.tokenizer.decode(query_txt_without_prompt['input_ids'])
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])
        
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        # query = _prepare_data_dict(query_txt_without_prompt, query_img_path, image_path_prefix)
        instance = {"query": query}
        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
            self.image_path_prefix,
        )
        instance.update({"pos_cand": pos_cand})
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        query_message = self.construct_messages(query_dict)
        cand_message = self.construct_messages(cand_dict)

        return query_message, cand_message 

class QueryDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
    ) -> None:
        super(QueryDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix = image_path_prefix 

    def __len__(self) -> int:
        return len(self.query_data)

    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'txt' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
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
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 

        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(f"{query_txt}")

        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        # query = _prepare_data_dict(query_txt_without_prompt, query_img_path, image_path_prefix)
        instance = {"query": query}
        instance['query']['qid'] = hash_qid(qid)
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query = instance['query']
        qid = query['qid']
        query_message = self.construct_messages(query)
        
        return query_message, qid 


class CandidateDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str, 
    ) -> None:
        super(CandidateDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix = image_path_prefix 

    def __len__(self) -> int:
        return len(self.cand_pool)

    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'txt' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
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
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)
        img = _load_and_preprocess_image(img_path, self.image_path_prefix)
        did = mbeir_cand_pool_entry.get("did", None)
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", None)
        if img is not None and cand_txt != '':
            instance = {
                "txt": cand_txt,
                "image": img, 
                "modality": cand_modality,
            }
        elif img is not None:
            instance = {
                "image": img, 
                "modality": cand_modality,
            }
        else:
            instance = {
                "txt": cand_txt,
                "modality": cand_modality,
            }
        instance.update({"did": hash_did(did)})
        return instance 
    

    def __getitem__(self, i):
        candidate = self.get_instance(i)
        did = candidate['did']
        candidate_message = self.construct_messages(candidate)
        
        return candidate_message, did 


def _load_data(data_path):
    """Validate and load data."""
    assert os.path.exists(data_path), f"Data Path {data_path} does not exist"
    assert data_path.endswith(".jsonl"), f"Data Path {data_path} is not a jsonl file"
    data_entries = _load_data_jsonl(data_path)
    return data_entries

def _load_query_data(query_data_path):
    query_data = _load_data(query_data_path)
    return query_data

def _load_cand_pool_as_dict(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    cand_pool_dict = {}
    for cand_pool_entry in cand_pool:
        did = cand_pool_entry.get("did")
        assert did, f"Cannot find did for {cand_pool_entry}"
        cand_pool_dict[did] = cand_pool_entry
    cand_pool = cand_pool_dict
    return cand_pool 

def _load_query_instructions(instructions_path):
    """Validate and load instructions."""
    # Validate the path and file extension
    assert os.path.exists(instructions_path), f"Instructions Path {instructions_path} does not exist"
    assert instructions_path.endswith(".tsv"), f"Instructions Path {instructions_path} is not a tsv file"
    prompts_dict = {}
    with open(instructions_path, "r") as f:
        next(f)  # Skip the header line
        for line in f.readlines():
            parts = line.strip().split("\t")
            # Construct the key to be dataset_id, query_modality, cand_modality
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
            prompts_dict[key] = prompts
    query_instructions = prompts_dict
    return query_instructions 

def _get_random_cand(cand_list):
    return random.choice(cand_list)

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s

def _get_random_query_prompt(dataset_id, query_modality, cand_modality, query_instructions):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = query_instructions.get(key, [])
    assert prompts, f"Cannot find prompts for {key}"
    prompt = format_string(random.choice(prompts))
    assert prompt, f"Prompt is empty for {key}"
    return prompt

def _load_and_preprocess_image(query_img_path, image_path_prefix):
    """Load an image given a path"""
    if not query_img_path:
        return None
    full_query_img_path = os.path.join(image_path_prefix, query_img_path)
    assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
    return full_query_img_path

def _prepare_data_dict(txt, img_path, image_path_prefix):
    img = _load_and_preprocess_image(img_path, image_path_prefix)
    if img is None:
        return {'txt': txt}
    elif txt == '':
        return {'image': img}
    return {"txt": txt, "image": img}

def _load_data_jsonl(datapath):
    data_entries = []
    with open(datapath, "r") as fin:
        for line in fin:
            data_entry = json.loads(line)
            data_entries.append(data_entry)
    return data_entries

def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def _load_cand_pool(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    return cand_pool
