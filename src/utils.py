from src.config import Config

import numpy as np

class Utils:
    @staticmethod
    def set_seed(seed=None):
        if seed is None:
            seed = Config.SEED
        np.random.seed(seed)

    @staticmethod
    def generate_data(embd_dim: int, num_queries: int, num_docs: int):
        # TODO: Implement more complicated generation methods

        # Set random seed for reproducibility
        Utils.set_seed()

        # Generate random queries
        queries = np.random.random((num_queries, embd_dim)).astype('float32')
        queries[:, 0] += np.arange(num_queries) / 1000

        # Generate random documents
        docs = np.random.random((num_docs, embd_dim)).astype('float32')
        docs[:, 0] += np.arange(num_docs) / 1000

        return queries, docs

    @staticmethod
    def l2_distance(queries, keys):
        """
        Calculate L2 distance between queries and keys.
        queries: (num_queries, embd_dim)
        keys: (num_keys, embd_dim)
        """

        # # Broadcasting: (num_queries, 1, embd_dim) - (1, num_keys, embd_dim)
        # squares = queries[:, np.newaxis] - keys
        # # Got a (num_queries, num_keys, embd_dim) array.
        # distances = np.linalg.norm(squares, axis=2)

        queries_norm = np.sum(queries**2, axis=1).reshape(-1, 1)
        keys_norm = np.sum(keys**2, axis=1)
        squares = queries_norm + keys_norm - 2 * np.dot(queries, keys.T)
        squares = np.clip(squares, a_min=0, a_max=None)
        distances = np.sqrt(squares)

        return distances

    @staticmethod
    def get_recall(ground_truths, predicts, axis=None):
        assert ground_truths.shape[0] == predicts.shape[0]

        recall = np.mean([
            np.isin(ground_truth, predict) 
            for ground_truth, predict in zip(ground_truths, predicts)
        ], axis=axis)

        return recall