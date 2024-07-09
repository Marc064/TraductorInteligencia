import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data.Spanish
        self.target_text = self.data.English

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, padding='max_length', truncation=True, return_tensors="pt"
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, padding='max_length', truncation=True, return_tensors="pt"
        )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }

def load_data(tokenizer, source_len, target_len, train_batch_size, valid_batch_size):
    df = pd.read_csv('data/spanish-to-english.csv')

    valid_data = df.sample(frac=0.5, random_state=42)
    train_data = df.drop(valid_data.index).reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    train_dataset = TranslationDataset(train_data, tokenizer, source_len, target_len)
    valid_dataset = TranslationDataset(valid_data, tokenizer, source_len, target_len)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader
