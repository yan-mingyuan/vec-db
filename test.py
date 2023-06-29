from src.config import Config
from src.utils import Utils
from src.engine import VectorEngine

from time import time
import numpy as np

def perform_search(db, queries, brute, symmetric=False, cached=True, top_k=5):
    start_time = time()
    results = db.search(queries, brute=brute, symmetric=symmetric, cached=cached, top_k=top_k)
    search_time = time() - start_time
    return results, search_time

embd_dim = 120
num_queries = 1000
num_docs = 100_0000
queries, docs = Utils.generate_data(embd_dim, num_queries, num_docs)
codebook_size = int(np.floor(np.sqrt(num_docs))) // 2
num_subvectors = 3

queries_topk = 5
keys_topk = 20

db = VectorEngine(embd_dim)
db.add(docs)
db.indexing(method='vectored', codebook_size=codebook_size)

# Perform ground truth search
ground_truths, search_time_gt = perform_search(db, queries, brute=True, top_k=queries_topk)
print("Time:", search_time_gt)

print(Config.SPLIT)

# Perform asymmetric vector quantization search
vq_asym_results, search_time_vq_asym = perform_search(db, queries, brute=False, symmetric=False, top_k=keys_topk)
search_recall_vq_asym = Utils.get_recall(ground_truths, vq_asym_results)
print("Time:", search_time_vq_asym)
print("VQ asymmetric recall:", search_recall_vq_asym)

# Perform symmetric vector quantization search
vq_sym_results, search_time_vq_sym = perform_search(db, queries, brute=False, symmetric=True, top_k=keys_topk)
search_recall_vq_sym = Utils.get_recall(ground_truths, vq_sym_results)
print("Time:", search_time_vq_sym)
print("VQ symmetric recall:", search_recall_vq_sym)

print(Config.SPLIT)

db.indexing(method='producted', codebook_size=codebook_size, num_subvectors=num_subvectors)

# Perform asymmetric vector quantization search
pq_asym_results, search_time_pq_asym = perform_search(db, queries, brute=False, symmetric=False, cached=False, top_k=keys_topk)
search_recall_pq_asym = Utils.get_recall(ground_truths, pq_asym_results)
print("Time:", search_time_pq_asym)
print("PQ asymmetric recall:", search_recall_pq_asym)

pq_asym_results, search_time_pq_asym = perform_search(db, queries, brute=False, symmetric=False, cached=True, top_k=keys_topk)
search_recall_pq_asym = Utils.get_recall(ground_truths, pq_asym_results)
print("Time:", search_time_pq_asym)
print("PQ asymmetric recall:", search_recall_pq_asym)

# Perform symmetric vector quantization search
pq_sym_results, search_time_pq_sym = perform_search(db, queries, brute=False, symmetric=True, top_k=keys_topk)
search_recall_pq_sym = Utils.get_recall(ground_truths, pq_sym_results)
print("Time:", search_time_pq_sym)
print("PQ symmetric recall:", search_recall_pq_sym)
