from bm25_utils import BM25Gensim
from text_utils import preprocess
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
import numpy as np
from ranking import Ranking
pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=10)
from config import (
        BM25_PATH, 
        WIKI_PATH, 
        TEST_PATH, 
        WEIGHT_MODEL, 
        MODEL_NAME
    )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ranker = Ranking(
    model_name=MODEL_NAME
)
ranker.load_model(WEIGHT_MODEL)

def metrics(bm25_model: BM25Gensim, list_query: list, list_ans: list, topk: int) -> float:
    out = []
    for query, ans in tqdm(list(zip(list_query, list_ans)), desc='run'):
        query = preprocess(query).lower()
        top_n_BM25, bm25_scores = bm25_model.get_topk(query, topk=1500)
        texts = [preprocess(df_wiki_windows.text_clean.values[i]) for i in top_n_BM25]
        question = preprocess(query)
        ranking_preds = ranker.get_score(question, texts).cpu().numpy()
        ranking_scores = ranking_preds*bm25_scores
        best_idxs = np.argsort(ranking_scores)[::-1]
        ranking_scores = np.array(ranking_scores)[best_idxs]
        top_n = np.array(top_n_BM25)[best_idxs]
        out.append((int(ans), top_n_BM25, top_n))
    count = 0
    for ans, top_n_BM25, top_n in out:
        if ans in top_n[:topk]:
            count+=1
    return count/len(out)

# Load model and wiki
bm25_model = BM25Gensim(BM25_PATH)
df_wiki_windows = pd.read_csv(WIKI_PATH)
df_wiki = df_wiki_windows.fillna("NaN")

# Load data test
df_test = pd.read_csv(TEST_PATH)


RECALL_NUM = 10
out = metrics(
    bm25_model = bm25_model, 
    list_query = df_test['question'], 
    list_ans = df_test['index'], 
    topk = RECALL_NUM)
print(f"Recall@{RECALL_NUM}: {out}")