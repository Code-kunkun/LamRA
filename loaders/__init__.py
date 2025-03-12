LOADERS = {}

def register_loader(name):
    def register_loader_cls(cls):
        if name in LOADERS:
            return LOADERS[name]
        LOADERS[name] = cls
        return cls
    return register_loader_cls

from .qwen2_5_vl import Qwen2_5VLModelLoader