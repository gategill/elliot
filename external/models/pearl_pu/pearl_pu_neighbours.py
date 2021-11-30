import pickle

from icecream import ic
ic.configureOutput(includeContext=True)

import numpy as np
from scipy import sparse
#from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, manhattan_distances
#from sklearn.metrics import pairwise_distances


class SelectNeighbours(object):
    """
    Simple kNN class
    """

    def __init__(self, data, params):
        ic()
        # deep copy or not?
        self._data = data
        #self._ratings = data.train_dict
        
        
        #self._num_neighbors = num_neighbors
        self._strategy = params.strategy
        
        # is this our local deep copy?
        self._users = self._data.users
        self._items = self._data.items
        #self._private_users = self._data.private_users
        #self._public_users = self._data.public_users
        #self._private_items = self._data.private_items
        #self._public_items = self._data.public_items
        #self.initialize()

    def initialize(self):
        ic()

        self.supported_strategies = ["BS", "BS+", "SS", "CS", "CS+"]
        print(f"\nSupported Strategies: {self.supported_strategies}")
        
        self._similarity_matrix = np.empty((len(self._users), len(self._users)))
        self.process_strategy(self._strategy)


    def process_strategy(self, strategy):
        ic()
        
        if strategy == "BS":
            self._similarity_matrix = self.baseline_strategy()
            
        elif strategy == "BS+":
            self._similarity_matrix = self.baseline_strategy_with_overlap()
            
        elif strategy == "SS":
            self._similarity_matrix = self.similarity_strategy()
            
        elif strategy == "CS":
            self._similarity_matrix = self.combination_strategy()
            
        elif strategy == "CS+":
            self._similarity_matrix = self.combination_strategy_with_overlap()
            
        else:
            raise ValueError("Process Strategy: value for parameter 'strategy' not recognized."
                             f"\nAllowed values are: {self.supported_strategies}"
                             f"\nPassed value was {strategy}\n")
            
    def baseline_strategy(self, k = 10):
        pass
    
    def baseline_strategy_with_overlap(self, k = 10, phi = 10):
        pass
    
    def similarity_strategy(self, k_prime = 10):
        pass
    
    
    def combination_strategy(self, k = 10, k_prime = 10):
        pass
    
    def combination_strategy_with_overlap(self, k = 10, k_prime = 10, phi = 10 ):
        pass

    def select_neighbours(self, user_x_id, item_id):
        pass
    
    def get_user_recs(self):
        pass
 