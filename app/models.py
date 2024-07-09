import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.to(device)
