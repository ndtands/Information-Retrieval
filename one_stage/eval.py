from bm25_utils import BM25Gensim
from text_utils import preprocess
from utils import post_process
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=10)
from config import BM25_PATH, WIKI_PATH, TEST_PATH

def metrics(bm25_model: BM25Gensim, list_query: list, list_ans: list, topk: int) -> float:
    out = []
    for query, ans in tqdm(list(zip(list_query, list_ans)), desc='run'):
        query = preprocess(query).lower()
        top_n, bm25_scores = bm25_model.get_topk(query, topk=topk)
        out.append((int(ans), top_n, bm25_scores))
    count = 0
    for ans, indexes, _ in out:
        if ans in indexes.tolist()[:topk]:
            count+=1
    return count/len(out)

# Load model and wiki
bm25_model = BM25Gensim(BM25_PATH)
df_wiki_windows = pd.read_csv(WIKI_PATH)
df_wiki = df_wiki_windows.fillna("NaN")

# Load data test
df_test = pd.read_csv(TEST_PATH)


RECALL_NUM = 100
out = metrics(
    bm25_model = bm25_model, 
    list_query = df_test['question'], 
    list_ans = df_test['index'], 
    topk = RECALL_NUM)
print(f"Recall@{RECALL_NUM}: {out}")