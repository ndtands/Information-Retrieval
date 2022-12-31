import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity

class BM25Gensim:
    def __init__(self, checkpoint_path):
        self.dictionary = Dictionary.load(checkpoint_path + "/dict")
        self.tfidf_model = SparseMatrixSimilarity.load(checkpoint_path + "/tfidf")
        self.bm25_index = TfidfModel.load(checkpoint_path + "/bm25_index")
    
    def get_topk(self, query, topk=100):
        tokenized_query = query.split()
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(tokenized_query)]
        scores = self.bm25_index[tfidf_query]
        top_n = np.argsort(scores)[::-1][:topk]
        return top_n, scores[top_n]