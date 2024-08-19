from torchtitan.datasets import build_tokenizer
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

import torch
import numpy as np
model_name = "llama3"
model_flavor = "500M"

ckpt_path = "./outputs/checkpoint_500M/final_model.pt"
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

python evaluate.py \
    --model_name llama3 \
    --model_flavor 500M \
    --ckpt_path ./outputs/checkpoint_500M/final_model.pt \
    --tokenizer_path ./test/assets/tokenizer.model \
    --pretrained_embedding_path ./test/assets/llama3_8b_pretrained_embedding.npy

python evaluate.py \
    llama3 \
    500M \
    ./outputs/checkpoint_500M/final_model.pt \
    ./test/assets/tokenizer.model \
    ./test/assets/llama3_8b_pretrained_embedding.npy

python evaluate.py \
    llama3_pretrained \
    debugmodel \
    ./outputs/checkpoint_debug_pretrained/final_model.pt \
    ./test/assets/tokenizer.model \
    ./test/assets/llama3_8b_pretrained_embedding.npy

