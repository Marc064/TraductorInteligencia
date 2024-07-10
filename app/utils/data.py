import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import T5Tokenizer

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = str(self.data.iloc[index, 1])
        target_text = str(self.data.iloc[index, 2])

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.target_len, padding='max_length', truncation=True, return_tensors='pt')

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
    df = pd.read_csv('data/translations.csv')

    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=42)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALIDATION Dataset: {}".format(val_dataset.shape))

    training_set = TranslationDataset(train_dataset, tokenizer, source_len, target_len)
    val_set = TranslationDataset(val_dataset, tokenizer, source_len, target_len)

    train_params = {'batch_size': train_batch_size, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': valid_batch_size, 'shuffle': False, 'num_workers': 0}

    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    return train_loader, val_loader
