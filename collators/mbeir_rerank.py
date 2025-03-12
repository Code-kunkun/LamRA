from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_5_vision_process import process_vision_info

def extract_inputs(inputs, PAD_TOKEN_ID, IGNORE_TOKEN_ID):
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

    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    labels[labels == PAD_TOKEN_ID] = IGNORE_TOKEN_ID

    return input_ids, labels, attention_mask, pixel_values, image_grid_thw 

class MbeirRerankDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # training stage, set the tokenizer padding side to right
        self.processor.tokenizer.padding_side = 'right'

        category_size = len(messages[0])
        
        rerank_messages = []

        for category in range(category_size):
            for item in messages:
                rerank_messages.append(item[category])

        rerank_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in rerank_messages
        ]
        rerank_image_inputs, rerank_video_inputs = process_vision_info(rerank_messages)
        rerank_inputs = self.processor(
            text=rerank_texts,
            images=rerank_image_inputs,
            videos=rerank_video_inputs,
            padding=True,
            return_tensors="pt"
        )

        rerank_input_ids, rerank_labels, rerank_attention_mask, rerank_pixel_values, rerank_image_grid_thw = extract_inputs(rerank_inputs, self.PAD_TOKEN_ID, self.IGNORE_TOKEN_ID)
        # additional adjustments to the mask are required here for reranking as it involves the computation of NTP loss.
        # set the user provided info to -100
        for i in range(len(rerank_labels)):
            start_indices = (rerank_input_ids[i] == 151644).nonzero(as_tuple=True)[0]
            for j in range(1, len(start_indices), 2):
                idx1, idx2 = start_indices[j], start_indices[j + 1]
                rerank_labels[i][idx1:idx2] = self.IGNORE_TOKEN_ID
        # set the system prompt to -100
        rerank_labels[:, :11] = self.IGNORE_TOKEN_ID
        
        return dict(
            input_ids=rerank_input_ids,
            attention_mask=rerank_attention_mask,
            pixel_values=rerank_pixel_values,
            image_grid_thw=rerank_image_grid_thw,
            labels=rerank_labels,
        )