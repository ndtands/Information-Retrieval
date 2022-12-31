from ranking.pairwise_model import (
    PairwiseModel,
    SiameseDataset,
    DataLoader, 
    collate_fn
)
from transformers import AutoTokenizer
import pandas as pd
import typing
import torch

class Ranking:
    def __init__(self, model_name: str, max_length: int=384, batch_size: int=64) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =PairwiseModel(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.max_length = max_length
        self.batch_size = batch_size
        
    def load_model(self, path: str) -> PairwiseModel:
        self.model.load_state_dict(torch.load(path))

    def get_score(self, question: str, texts: typing.List)-> typing.List:
        self.model.to(self.device)
        self.model.eval()
        tmp = pd.DataFrame()
        tmp["text"] = [" ".join(x.split()) for x in texts]
        tmp["question"] = question
        valid_dataset = SiameseDataset(tmp, self.tokenizer, self.max_length, is_test=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                    num_workers=0, shuffle=False, pin_memory=True)   
        preds = []
        with torch.no_grad():
            bar = enumerate(valid_loader)
            for _, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                preds.append(torch.sigmoid(self.model(ids, masks)).view(-1))
            preds = torch.concat(preds)
        return preds