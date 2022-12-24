from nltk import word_tokenize as lib_tokenizer 
from transformers import AutoTokenizer
from configs import MODEL_NAME
import numpy as np
import re 
import torch
dict_map = dict({}) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pad_token_id = tokenizer.pad_token_id

def word_tokenize(text): 
    global dict_map 
    words = text.split() 
    words_norm = [] 
    for w in words: 
        if dict_map.get(w, None) is None: 
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"') 
        words_norm.append(dict_map[w]) 
    return words_norm 

def strip_answer_string(text): 
    text = text.strip() 
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`': 
        if text[0] != '(' and text[-1] == ')' and '(' in text: 
            break 
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1: 
            break 
        text = text[:-1].strip() 
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`': 
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1: 
            break 
        text = text[1:].strip() 
    text = text.strip() 
    return text 
 
def strip_context(text): 
    text = text.replace('\n', ' ') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    return text

def collate_fn(batch):
    ids = [torch.cat([x["ids1"], x["ids2"]]) for x in batch]
    targets = [x["target"] for x in batch]
    max_len = np.max([len(x) for x in ids])
    masks = []
    for i in range(len(ids)):
        if len(ids[i]) < max_len:
            ids[i]= torch.cat((ids[i], torch.tensor([pad_token_id,]*(max_len - len(ids[i])),dtype=torch.long)))
        masks.append(ids[i] != pad_token_id)
    # print(tokenizer.decode(ids[0]))
    outputs = {
        "ids": torch.vstack(ids),
        "masks": torch.vstack(masks),
        "target": torch.vstack(targets).view(-1)
    }
    return outputs