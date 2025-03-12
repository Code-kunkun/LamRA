from typing import Tuple

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from .base import BaseModelLoader


class Qwen2_5VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            ) 

        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        return model, tokenizer, processor 