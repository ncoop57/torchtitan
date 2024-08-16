from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

model_name = "llama3"
model_flavor = "70b"
ckpt_path = f"checkpoints/{model_name}/{model_flavor}/model.pt"
model_cls = model_name_to_cls[model_name]
model_config = models_config[model_name][model_flavor]
tokenizer_name = model_name_to_tokenizer[model_name]
tokenizer = build_tokenizer(tokenizer_name)
model = model_cls.from_model_args(model_config)

# load model
model.load_state_dict(torch.load(ckpt_path))


