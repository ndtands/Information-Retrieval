import regex as re
import string
from nltk import word_tokenize as lib_tokenizer 

def post_process(x: str) -> str:
    x = x.replace('BULLET : : : :',"")
    x = x.replace('bullet : : : :',"")
    x = x.replace('===',"")
    x = x.replace('==',"")
    x = x.replace('; ;',"")
    x = " ".join(word_tokenize(strip_context(x))).strip()
    x = x.replace("\n"," ")
    x = "".join([i for i in x if i not in string.punctuation])
    return x

dict_map = dict({})  
def word_tokenize(text: str) -> str: 
    global dict_map 
    words = text.split() 
    words_norm = [] 
    for w in words: 
        if dict_map.get(w, None) is None: 
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"') 
        words_norm.append(dict_map[w]) 
    return words_norm 
 
def strip_answer_string(text: str) -> str: 
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
 
def strip_context(text: str) -> str: 
    text = text.replace('\n', ' ') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    return text
