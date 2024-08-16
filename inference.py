from torchtitan.datasets import build_tokenizer
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

import torch

model_name = "llama3"
model_flavor = "debugmodel"
ckpt_path = f"checkpoints/{model_name}/{model_flavor}/model.pt"
tokenizer_path = "./test/assets/test_tiktoken.model"
model_cls = model_name_to_cls[model_name]
model_config = models_config[model_name][model_flavor]
tokenizer_name = model_name_to_tokenizer[model_name]
tokenizer = build_tokenizer(tokenizer_name, tokenizer_path)
model_config.vocab_size = tokenizer.n_words
model = model_cls.from_model_args(model_config)
# model = model_cls.from_model_args(model_config)
print(tokenizer.encode("Hello, world!", bos=True, eos=True))
print(model)
print(model.generate("Hello, world!", tokenizer))

# load model
# model.load_state_dict(torch.load(ckpt_path))


