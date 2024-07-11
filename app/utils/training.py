import os
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from app.utils.checkpoint import save_checkpoint, load_checkpoint

SOURCE_LEN = 128
TARGET_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2


class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data['Spanish']
        self.target_text = self.data['English']

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True, truncation=True, return_tensors='pt'
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, pad_to_max_length=True, truncation=True, return_tensors='pt'
        )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }


def train_model():
    # Load the dataset
    df = pd.read_csv('data/translations.csv')

    # Split the dataset into training and validation sets
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare the training and validation datasets
    train_dataset = TranslationDataset(train_df, tokenizer, source_len=SOURCE_LEN, target_len=TARGET_LEN)
    val_dataset = TranslationDataset(val_df, tokenizer, source_len=SOURCE_LEN, target_len=TARGET_LEN)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # Define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    best_valid_loss = float('inf')
    patience_counter = 0

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        start_epoch, last_loss = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()

            input_ids = batch['source_ids'].to(device, dtype=torch.long)
            attention_mask = batch['source_mask'].to(device, dtype=torch.long)
            labels = batch['target_ids'].to(device, dtype=torch.long)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training Loss for Epoch {epoch + 1}: {avg_train_loss}")

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                input_ids = batch['source_ids'].to(device, dtype=torch.long)
                attention_mask = batch['source_mask'].to(device, dtype=torch.long)
                labels = batch['target_ids'].to(device, dtype=torch.long)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(val_loader)
        print(f"Validation Loss for Epoch {epoch + 1}: {avg_valid_loss}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_checkpoint(model, optimizer, epoch, avg_valid_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break
