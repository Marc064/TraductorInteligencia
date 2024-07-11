import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.to(device)

# Cargar el checkpoint más reciente
checkpoint_dir = 'checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
