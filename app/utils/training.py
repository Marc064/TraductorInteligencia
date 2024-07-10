import os
import torch
from tqdm import tqdm
from app.models import model, tokenizer, device
from app.utils.data import load_data
from app.utils.checkpoint import save_checkpoint, load_checkpoint

SOURCE_LEN = 128
TARGET_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2

def train_model():
    train_loader, valid_loader = load_data(tokenizer, SOURCE_LEN, TARGET_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE)

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

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
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
        print(f"Training Loss for Epoch {epoch+1}: {avg_train_loss}")

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}"):
                input_ids = batch['source_ids'].to(device, dtype=torch.long)
                attention_mask = batch['source_mask'].to(device, dtype=torch.long)
                labels = batch['target_ids'].to(device, dtype=torch.long)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Validation Loss for Epoch {epoch+1}: {avg_valid_loss}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_checkpoint(model, optimizer, epoch, avg_valid_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
