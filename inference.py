from fastcore.script import *

from torchtitan.datasets import build_tokenizer
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

import torch
import numpy as np

def load_model(model_name, model_flavor, ckpt_path, tokenizer_path, pretrained_embedding_path):
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][model_flavor]
    tokenizer_name = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_path)
    model_config.vocab_size = tokenizer.n_words
    if model_name.endswith("_pretrained"):
        model = model_cls.from_model_args_and_embedding(
            model_config,
            pretrained_embedding=np.load(pretrained_embedding_path)
        )
    else:
        model = model_cls.from_model_args(model_config)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    return model

@call_parse
def main(
    model_name: str,                    # name of the model [llama2, llama3, llama3_pretrained]
    model_flavor: str,                  # name of the model flavor [debugmodel, 500M]
    ckpt_path: str,                     # path to the checkpoint
    tokenizer_path: str,                # path to the tokenizer
    pretrained_embedding_path: str,     # path to the pretrained embedding
):
    model = load_model(model_name, model_flavor, ckpt_path, tokenizer_path, pretrained_embedding_path)
    print(model)

model_name = "llama3"
model_flavor = "debugmodel"

ckpt_path = "./outputs/checkpoint_debug/final_model.pt"
tokenizer_path = "./test/assets/tokenizer.model"
pretrained_embedding_path = "./test/assets/llama3_8b_pretrained_embedding.npy"
model_cls = model_name_to_cls[model_name]
model_config = models_config[model_name][model_flavor]
tokenizer_name = model_name_to_tokenizer[model_name]
tokenizer = build_tokenizer(tokenizer_name, tokenizer_path)
model_config.vocab_size = tokenizer.n_words
if model_name.endswith("_pretrained"):
    model = model_cls.from_model_args_and_embedding(model_config, pretrained_embedding=np.load(pretrained_embedding_path))
else:
    model = model_cls.from_model_args(model_config)
print(tokenizer.encode("Hello, world!", bos=True, eos=True))
print(model)
print(model.generate("Hello, world!", tokenizer))

# load model
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict, strict=False)
print(model.generate("Hi, my name is", tokenizer, top_k=50))

# print(model.generate("```python\nprint('Hello, ", tokenizer))

