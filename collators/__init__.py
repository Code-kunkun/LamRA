COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

from .qwen2_5_vl import Qwen2_5VLDataCollator