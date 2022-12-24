from torch.utils.data import Dataset
import torch
from pandas import DataFrame
from transformers import AutoTokenizer
from typing import Dict

class SiameseDataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: AutoTokenizer, max_length: int):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.content1 = tokenizer.batch_encode_plus(list(df.question.apply(lambda x: x.replace("_"," ")).values), max_length=max_length, truncation=True)["input_ids"]
        self.content2 = tokenizer.batch_encode_plus(list(df.text.apply(lambda x: x.replace("_"," ")).values), max_length=max_length, truncation=True)["input_ids"]
        self.targets = self.df.label
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        return {
            'ids1': torch.tensor(self.content1[index], dtype=torch.long),
            'ids2': torch.tensor(self.content2[index][1:], dtype=torch.long),
            'target': torch.tensor(self.targets[index], dtype=torch.float)
        }