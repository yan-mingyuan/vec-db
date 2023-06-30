from src.config import QuantizedType
from src.utils import Utils

import numpy as np
from sklearn.cluster import KMeans


class VectorEngine:
    def __init__(self, embd_dim: int):
        self.embd_dim = embd_dim
        self.lazy_data = []
        self.structed_data = np.array([], dtype='float32')
        self.is_structured = True
        self.is_indexed = False
    
    def add(self, vector: np.ndarray):
        if vector.ndim == 1:
            # Reshape the vector to a 2-dimensional array
            self.lazy_data.append(vector[np.newaxis, :])
        elif vector.ndim == 2:
            self.lazy_data.append(vector)
        else:
            raise NotImplementedError("Vectors with dimensions other than 1 or 2 are not supported.")
        
        self.is_structured = False
    
    @property
    def data(self):
        """
        Get the structured data from the lazy data.
        """
        if not self.is_structured:
            # Concatenate the data in the list and convert it to a numpy array
            self.structed_data = np.concatenate(self.lazy_data, axis=0)
            assert self.structed_data.shape[1] == self.embd_dim

            self.lazy_data = list(self.structed_data)
            self.is_structured = True
        
        return self.structed_data

    def vector_quantization(self, codebook_size):
        # Apply vector quantization to the data
        self.codebook_size = codebook_size
        self.quantized_type = QuantizedType.VECTOR

        # Use K-means clustering to generate the codebook
        self.kmeans = KMeans(n_clusters=codebook_size, n_init=10, max_iter=300, tol=1e-4)
        indexes = self.kmeans.fit_predict(self.data)
        self.codebook = self.kmeans.cluster_centers_

        # Create inverted indexes
        self.inverted_indexes = {inverted_index: np.where(indexes == inverted_index)[0] for inverted_index in range(codebook_size)}

    def truncate_k_candidates_per_query(self, query, cluster_index, top_k: int):
        candidate_indices = self.inverted_indexes[cluster_index]
        num_candidates = len(candidate_indices)

        if num_candidates < top_k:
            # If the number of candidates is less than top_k, pad the candidates with -1
            top_k_candidates_per_query = np.pad(candidate_indices, (0, top_k - num_candidates), constant_values=-1)
        else:
            # Retrieve the candidates from the database
            candidates = self.data[candidate_indices]

            # Calculate the distances between the query and candidates
            distances_per_query = Utils.l2_distance(query[np.newaxis, :], candidates)
            distances_per_query = distances_per_query.ravel()  # Flatten the distances array

            # Find the indices of the top-k nearest neighbors using argpartition
            top_k_candidates_per_query_arg = np.argpartition(distances_per_query, kth=top_k - 1, axis=0)[:top_k]
            top_k_candidates_per_query = candidate_indices[top_k_candidates_per_query_arg]

        return top_k_candidates_per_query

    def search_vector_quantization(self, queries, top_k=5, symmetric=False):
        top_k_candidates = []  # List to store the top-k candidates for each query
        codebook_indices = self.kmeans.predict(queries)

        if symmetric:
            # Less accurate but faster with cache

            # Cached results for each cluster
            cached_candidates = {}
            for cluster_index in range(self.codebook_size):
                # Precalculate and cache the top-k candidates for each centroid
                top_k_candidates_per_centroid = self.truncate_k_candidates_per_query(self.codebook[cluster_index], cluster_index, top_k)
                cached_candidates[cluster_index] = top_k_candidates_per_centroid

            # Retrieve the cached top-k candidates for each query
            for codebook_index in codebook_indices:
                top_k_candidates.append(cached_candidates[codebook_index])

        else:
            # More accurate but slower

            # Calculate the top-k candidates for each query
            for query, cluster_index in zip(queries, codebook_indices):
                top_k_candidates_per_query = self.truncate_k_candidates_per_query(query, cluster_index, top_k)
                top_k_candidates.append(top_k_candidates_per_query)

        return np.array(top_k_candidates)

    def product_quantization(self, codebook_size, num_subvectors):
        assert self.embd_dim % num_subvectors == 0

        # Apply product quantization to the data
        self.codebook_size = codebook_size
        self.num_subvectors = num_subvectors
        self.quantized_type = QuantizedType.PRODUCT

        # Split the vectors into subvectors
        # not move physically, just reshape to speed up
        self.subvectors = self.data.reshape(-1, self.embd_dim // num_subvectors)

        # Use K-means clustering on subvectors
        self.kmeans = KMeans(n_clusters=codebook_size, n_init=10, max_iter=300, tol=1e-4)
        indexes = self.kmeans.fit_predict(self.subvectors)
        quantized_keys_indexes = indexes.reshape(-1, num_subvectors)
        self.sub_keys_indexes_partitions = quantized_keys_indexes.T
        self.codebook = self.kmeans.cluster_centers_

    def search_product_quantization(self, queries, top_k=5, symmetric=False, cached=True):
        sub_queries = queries.reshape(-1, self.embd_dim // self.num_subvectors)
        coarse_distances_partitions = []

        if symmetric:
            # Less accurate but faster with cache

            # Cache paired distances between codebook entries
            cached_paired_coarse_distances = Utils.l2_distance(self.codebook, self.codebook)

            # Quantize sub-queries using the codebook
            codebook_indices = self.kmeans.predict(sub_queries).reshape(-1, self.num_subvectors)
            codebook_indices_partitions = np.split(codebook_indices, self.num_subvectors, axis=1)
            for (codebook_indices_partition, sub_keys_indexes_partition) in zip(codebook_indices_partitions, self.sub_keys_indexes_partitions):
                codebook_indices_partition = codebook_indices_partition.squeeze()
                # Retrieve coarse distances using the cached paired distances
                coarse_distances_partition = cached_paired_coarse_distances[codebook_indices_partition[:, np.newaxis], sub_keys_indexes_partition]
                coarse_distances_partitions.append(coarse_distances_partition)

        else:
            # More accurate but slower
            
            if cached:
                # Cache distances for every query
                num_queries = len(queries)
                cached_coarse_distances = Utils.l2_distance(sub_queries, self.codebook).reshape(num_queries, self.num_subvectors, -1)

                # Split cached coarse distances into partitions
                cached_coarse_distances_partitions = np.split(cached_coarse_distances, self.num_subvectors, axis=1)
                for (cached_coarse_distances_partition, sub_keys_indexes_partition) in zip(cached_coarse_distances_partitions, self.sub_keys_indexes_partitions):
                    cached_coarse_distances_partition = cached_coarse_distances_partition.squeeze()
                    # Retrieve coarse distances for each sub-query using the cached distances
                    coarse_distances_partition = [
                        cached_coarse_distance_partition[sub_keys_indexes_partition]
                        for cached_coarse_distance_partition in cached_coarse_distances_partition
                    ]
                    coarse_distances_partitions.append(coarse_distances_partition)
            else:
                # Without cache
                sub_queries_partitions = np.split(queries, self.num_subvectors, axis=1)
                coarse_distances_partitions = []
                for (sub_queries_partition, sub_keys_indexes_partition) in zip(sub_queries_partitions, self.sub_keys_indexes_partitions):
                    sub_keys_partition = self.codebook[sub_keys_indexes_partition]
                    # Calculate coarse distances between sub-queries and sub-codebooks
                    coarse_distances_partition = Utils.l2_distance(sub_queries_partition, sub_keys_partition)
                    coarse_distances_partitions.append(coarse_distances_partition)
                
                coarse_distances = np.sum(coarse_distances_partitions, axis=0)
                top_k_candidates = np.argpartition(coarse_distances, kth=top_k - 1, axis=1)[:, :top_k]

        # Aggregate coarse distances from different partitions
        coarse_distances = np.sum(coarse_distances_partitions, axis=0)
        # Find top-k candidates based on coarse distances
        top_k_candidates = np.argpartition(coarse_distances, kth=top_k - 1, axis=1)[:, :top_k]

        return top_k_candidates

    def indexing(self, method: str, codebook_size: int, num_subvectors: int = 1):
        # TODO: Implement more complicated indexing methods, like hierarchical-KMeans

        Utils.set_seed()

        if method == 'vectored':
            self.vector_quantization(codebook_size)
        elif method == 'producted':
            self.product_quantization(codebook_size, num_subvectors)
        else:
            raise NotImplementedError("Invalid quantized_type.")

    def search(self, queries, brute=False, symmetric=False, cached=True, top_k: int = 10):
        if brute:
            # brute-force compute all
            assert top_k <= len(self.data)
            
            distances = Utils.l2_distance(queries, self.data)
            top_k_candidates = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
            return top_k_candidates

        else:
            if self.quantized_type == QuantizedType.VECTOR:
                return self.search_vector_quantization(queries, top_k, symmetric)
            elif self.quantized_type == QuantizedType.PRODUCT:
                return self.search_product_quantization(queries, top_k, symmetric, cached)
            else:
                raise NotImplementedError("Invalid quantized_type.")