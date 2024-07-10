import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

