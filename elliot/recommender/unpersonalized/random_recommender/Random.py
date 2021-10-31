"""
Created on April 4, 2020
Tensorflow 2.1.0 implementation of APR.
@author Anonymized
"""

import numpy as np

from icecream import ic
ic.configureOutput(includeContext=True)


from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger


class Random(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ic()
        ic(self._data.transactions)

        """
        Create a Random recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """

        self._params_list = [
            ("_seed", "random_seed", "seed", 42, int, None)
        ]
        self.autoset_params()

        np.random.seed(self._seed)

    @property
    def name(self):
        return f"Random_{self.get_params_shortcut()}"

    def train(self):
        ic()
        self.evaluate()

    def get_recommendations(self, top_k: int = 100):
        ic()
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(top_k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, top_k, *args):
        ic()
        #ic(mask)
        r_int = np.random.randint
        n_items = self._num_items
        items = self._data.items
        ratings = self._data.train_dict
        #ic(ratings)
        
        with open("data/movielens_2k/ratings_for_random.txt", "w") as f:
            f.writelines(str(ratings[75]))

        r = {}
        
        #ic(ratings.items())
        
        for u, i_s in ratings.items():
            l = []
            
            # ui is a set of movie ids for uesr u
            ui = set(i_s.keys())
            lui = len(ui)
            local_k = min(top_k, n_items - lui)
            
            # public users? unseen movie ids???
            local_items = np.arange(n_items)[mask[self._data.public_users[u]]]
            n_local_items = len(local_items)
            #ic(n_local_items)
            
            with open("data/movielens_2k/local_items.txt", "w") as f:
                f.writelines(str(local_items))


            for index in range(local_k):
                
                # private items?
                #j = self._data.private_items[local_items[r_int(n_local_items)]]
                j = items[r_int(n_items)]
                while j in ui:
                    #j = self._data.private_items[local_items[r_int(n_local_items)]]
                    j = items[r_int(n_items)]
                
                random_rating = np.random.randint(0.5, 5)
                l.append((j, random_rating)) # score is 1?
            r[u] = l
            
        with open("data/movielens_2k/random_r.txt", "w") as f:
            f.writelines(str(r))
        
        return r
