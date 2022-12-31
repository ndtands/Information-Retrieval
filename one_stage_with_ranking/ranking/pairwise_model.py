import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
import pandas as pd
import typing
from torch import Tensor
tokenizer = AutoTokenizer.from_pretrained('nguyenvulebinh/vi-mrc-base', use_auth_token=True)
pad_token_id = tokenizer.pad_token_id


class PairwiseModel(nn.Module):
    def __init__(self, model_name: str):
        super(PairwiseModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=True)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 1)
        
    def forward(self, ids: Tensor, masks: Tensor) -> Tensor:
        out = self.model(input_ids=ids,
                           attention_mask=masks,
                           output_hidden_states=False).last_hidden_state
        out = out[:,0]
        outputs = self.fc(out)
        return outputs


class SiameseDataset(Dataset):

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.content1 = tokenizer.batch_encode_plus(list(df.question.values), max_length=max_length, truncation=True)[
            "input_ids"]
        self.content2 = tokenizer.batch_encode_plus(list(df.text.values), max_length=max_length, truncation=True)[
            "input_ids"]
        if not self.is_test:
            self.targets = self.df.label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> typing.Dict:
        return {
            'ids1': torch.tensor(self.content1[index], dtype=torch.long),
            'ids2': torch.tensor(self.content2[index][1:], dtype=torch.long),
            'target': torch.tensor(0) if self.is_test else torch.tensor(self.targets[index], dtype=torch.float)
        }

def collate_fn(batch):
    ids = [torch.cat([x["ids1"], x["ids2"]]) for x in batch]
    targets = [x["target"] for x in batch]
    max_len =  np.max([len(x) for x in ids])
    masks = []
    for i in range(len(ids)):
        if len(ids[i]) < 512:
            ids[i] = torch.cat((ids[i], torch.tensor([pad_token_id, ] * (512 - len(ids[i])), dtype=torch.long)))
            masks.append(ids[i] != pad_token_id)
        else:
            ids[i] = ids[i][:512]
            masks.append(torch.tensor([True]*512))  
    try:
        outputs = {
            "ids": torch.vstack(ids),
            "masks": torch.vstack(masks),
            "target": torch.vstack(targets).view(-1)
        }
        return outputs
    except:
        print([len(i) for i in ids])
        print([len(i) for i in masks])
        outputs = {
            "ids": torch.vstack(ids),
            "masks": torch.vstack(masks),
            "target": torch.vstack(targets).view(-1)
        }
        return outputs
    
