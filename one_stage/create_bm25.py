import pandas as pd
from pandarallel import pandarallel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from utils import post_process
from gensim.similarities import SparseMatrixSimilarity
from config import BM25_PATH, PATH_WIKI
pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=10)

# Load data wiki
df_wiki = pd.read_csv(PATH_WIKI)
df_wiki = df_wiki.fillna("NaN")

# Get corpus
corpus = [x.split() for x in df_wiki['bm25_text'].values]

# Create dictionary
dictionary = Dictionary(corpus)

# Init model 
bm25_model = OkapiBM25Model(dictionary=dictionary)
bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
bm25_index = SparseMatrixSimilarity(
        corpus=bm25_corpus, 
        num_docs=len(corpus), 
        num_terms=len(dictionary), 
        normalize_queries=False, 
        normalize_documents=False
        )
tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # Enforce binary weighting of queries

# Save model
dictionary.save(f"{BM25_PATH}/dict")
tfidf_model.save(f"{BM25_PATH}/tfidf")
bm25_index.save(f"{BM25_PATH}/bm25_index")