import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("code-kunkun/LamRA-Ret-Qwen2.5VL-7b", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True).cuda()
processor = AutoProcessor.from_pretrained("code-kunkun/LamRA-Ret-Qwen2.5VL-7b")
tokenizer = processor.tokenizer 

def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    assert len(emb_tokens) == num_new_tokens

    model.resize_token_embeddings(len(tokenizer))

    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids

def get_embed_feature(hidden_states, input_ids, embed_index):
    embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1)
    embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
    return embed_features

def qwen2_5_vl_process(messages, processor):
    texts = [
      processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
      for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs


add_embed_token(tokenizer, model)

image_message = [
  {
      "role": "user",
      "content": [
          {"type": "image", "image": "./demo.jpeg"},
          {"type": "text", "text": "Find an image caption describing the following everyday image."},
          {"type": "text", "text": f"\nSummarize above image and sentence in one word: "}
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

text_message1 = [
  {
      "role": "user",
      "content": [
          {"type": "text", "text": "a dog and a woman are playing on the bench.\nSummarize above sentence in one word: "},
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

text_message2 = [
  {
      "role": "user",
      "content": [
          {"type": "text", "text": "a dog.\nSummarize above sentence in one word: "},
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

image_messages = [image_message]
text_messages = [text_message1, text_message2]

image_inputs = qwen2_5_vl_process(image_messages, processor)
text_inputs = qwen2_5_vl_process(text_messages, processor)

with torch.no_grad():
    text_hidden_state = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    text_embeds = get_embed_feature(text_hidden_state, text_inputs['input_ids'], model.config.emb_token_ids[0])
    image_hidden_state = model(**image_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    image_embeds = get_embed_feature(image_hidden_state, image_inputs['input_ids'], model.config.emb_token_ids[0])

    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

print(image_embeds @ text_embeds.t())