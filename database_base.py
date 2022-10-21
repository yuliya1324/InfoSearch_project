import scipy
import numpy as np
import pickle
from pathlib import Path

class DataBase:
    def get_query(self, query: str):
        raise NotImplementedError()
    
    def count_similarity(self, query: str):
        query_vector = self.get_query(query)
        return np.dot(self.matrix_answers, query_vector.T).toarray()