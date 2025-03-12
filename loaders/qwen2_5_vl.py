from typing import Tuple

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from . import register_loader
from .base import BaseModelLoader
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration


@register_loader("qwen2_5-vl-7b")
class Qwen2_5VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True, pretrain=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model:
            model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            ) 
        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        self.add_embed_token(tokenizer, model)

        return model, tokenizer, processor 

    def add_embed_token(self, tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens

        model.resize_token_embeddings(len(tokenizer))

        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids