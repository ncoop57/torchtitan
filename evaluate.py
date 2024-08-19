from datasets import load_dataset
from fastcore.script import *
from torcheval.metrics.text import Perplexity
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
    model.to('cuda')
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset, batch_size=64):
    metric = Perplexity()
    for text in dataset['text']:
        # print(text)
        input_ids = tokenizer.encode(text, bos=True, eos=True)
        # print(input_ids)
        input_ids = torch.tensor(input_ids).to('cuda')
        # add batch dim
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)
        labels = input_ids[:, 1:]
        # print devices
        # print(output.device, labels.device)
        # convert logits to logprobs
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        metric.update(logprobs[:, :-1].to('cpu'), labels.to('cpu'))
    
    ppl = metric.compute().item()
    print(f"Perplexity: {ppl}")

@call_parse
def main(
    model_name: str,                    # name of the model [llama2, llama3, llama3_pretrained]
    model_flavor: str,                  # name of the model flavor [debugmodel, 500M]
    ckpt_path: str,                     # path to the checkpoint
    tokenizer_path: str,                # path to the tokenizer
    pretrained_embedding_path: str,     # path to the pretrained embedding
    dataset_name: str = 'HuggingFaceTB/smollm-corpus',                  # name of the dataset [wikitext2, enwik8]
    dataset_split: str = 'train',
):
    model, tokenizer = load_model(model_name, model_flavor, ckpt_path, tokenizer_path, pretrained_embedding_path)
    dataset = load_dataset(dataset_name, 'cosmopedia-v2', split=dataset_split).select(range(10_000))
    evaluate_model(model, tokenizer, dataset)