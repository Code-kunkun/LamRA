from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_5_vision_process import process_vision_info


class MbeirQueryDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        qids = []
        for item in messages:
            new_messages.append(item[0])
            qids.append(item[1])

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None 
        
        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            qids=qids
        )

class MbeirCandidateDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        dids = []

        for item in messages:
            new_messages.append(item[0])
            dids.append(item[1])

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
        # inputs = self.processor(
        #     text=texts,
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None 
        
        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            dids=dids
        )