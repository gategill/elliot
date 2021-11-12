"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Michal Polak Szarkowicz'
__email__ = '118304271@umail.ucc.ie'

import pickle
import time

#from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.utils.folder import build_model_folder

from icecream import ic
ic.configureOutput(includeContext=True)


#from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.user_knn.user_knn_similarity import Similarity
from elliot.recommender.knn.user_knn.aiolli_ferrari import AiolliSimilarity

from elliot.recommender.base_recommender_model import init_charger

from elliot.recommender.knn.user_knn import UserKNN



#class RecursiveUserKNN(UserKNN):
class RecUserKNN(UserKNN):
    r"""
    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type 'standard'

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        RecUserKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          implementation: standard
    """
    # should this be a simple init?
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        #UserKNN.__init__(self.data, config, params, *args, **kwargs)
        ic()

        self._params_list = [
            # variable name, public name, shortcut, default, type, ? 
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implementation", "implementation", "imp", "standard", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_shrink", "shrink", "shrink", 0, None, None),
            ("_normalize", "normalize", "norm", True, None, None),
            ("_asymmetric_alpha", "asymmetric_alpha", "asymalpha", False, None, lambda x: x if x else ""),
            ("_tversky_alpha", "tversky_alpha", "tvalpha", False, None, lambda x: x if x else ""),
            ("_tversky_beta", "tversky_beta", "tvbeta", False, None, lambda x: x if x else ""),
            ("_row_weights", "row_weights", "rweights", None, None, lambda x: x if x else "")
        ]
        self.autoset_params()

        #self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)

    # if the same then don't override!!!
   #def get_single_recommendation(self, mask, k, *args):
   #    ic()
   #    user_recs = {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}
   #    with open("data/movielens_2k/get_user_recs.txt", "w") as f:
   #        for u, recs in user_recs.items():
   #            f.writelines(str(u) + " : " + str(recs) + "\n\n")
   #    return user_recs



   #def get_recommendations(self, k: int = 10):
   #    ic()

   #    #WHY TWO???
   #    predictions_top_k_val = {}
   #    predictions_top_k_test = {}

   #    recs_val, recs_test = self.process_protocol(k)

   #    predictions_top_k_val.update(recs_val)
   #    predictions_top_k_test.update(recs_test)
   #    
   #    with open("data/movielens_2k/predictions_top_k_val_RECURSIVE.txt", "w") as f:
   #        for u, recs in predictions_top_k_val.items():
   #            f.writelines(str(u) + " : " + str(recs) + "\n\n")

   #    return predictions_top_k_val, predictions_top_k_test

    @property
    def name(self):
        ic()

        return f"RecUserKNN_{self.get_params_shortcut()}"

    def train(self):
        ic()

        #if self._restore:
        #    return self.restore_weights()
        
        # enrichment phase
        ENRICHMENTS = 0
        for i in range(ENRICHMENTS):
            ic(i)
            ic(self._data.transactions)
            ic("before Similarity")
            #model_instance = UserKNN(data = self._data, config = self._config, params=self._params)
            #model_instance = UserKNN(data = self._data, config = self._config, params=self._params)
            self._ratings = self._data.train_dict
            self.set_model()
            #self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)

            ic("before model_instance.ititiaize()")
            #model = self.model_class(data=data_obj, config=self.base, params=model_params)
            self._model.initialize()
            
            ic("before new_recs_df")
            new_recs_df = self.get_recommendations(df = True)
            ic("defore add new rect to train set")
            self._data.add_new_recs_to_train_set(new_recs_df)
            ic("Success for this roundS")
            
            del self._model
            del self._ratings
          
            
        self._ratings = self._data.train_dict
        self.set_model()
        #self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)
        
        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()

