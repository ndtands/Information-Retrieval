import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel

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