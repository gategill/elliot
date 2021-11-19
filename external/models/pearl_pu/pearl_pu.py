"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import time
import pandas as pd


import os
import matplotlib
import matplotlib.pyplot as plt


from icecream import ic
ic.configureOutput(includeContext=True)
import copy

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.user_knn.user_knn_similarity import Similarity
from elliot.recommender.knn.user_knn.aiolli_ferrari import AiolliSimilarity
from elliot.recommender.base_recommender_model import init_charger


class PearlPu(RecMixin, BaseRecommenderModel):
    r"""
    GroupLens: An Open Architecture for Collaborative Filtering of Netnews

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type ('aiolli', 'classical')

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        UserKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          implementation: aiolli
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # copy it self._data = data
        ic("DOING WEIRD COPY STUFF")
        ic(self._data.transactions)
        
        #self._data = copy.deepcopy(data)
        #ic(self._data.transactions)


        #self._data.add_new_recs_to_train_set()
        ic(self._data.transactions)

        ic()

        self._params_list = [
            # variable_name, public_name, shortcut, default, reading_function, _ 
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
        
        ic(self._params)

        self._ratings = self._data.train_dict
        self.set_model()
        
        #ic(self._data.train_dict[75])
        
    def set_model(self):
        ic()
        if self._implementation == "aiolli":
            self._model = AiolliSimilarity(data=self._data,
                                           maxk=self._num_neighbors,
                                           shrink=self._shrink,
                                           similarity=self._similarity,
                                           implicit=self._implicit,
                                           normalize=self._normalize,
                                           asymmetric_alpha=self._asymmetric_alpha,
                                           tversky_alpha=self._tversky_alpha,
                                           tversky_beta=self._tversky_beta,
                                           row_weights=self._row_weights)
        else:
            if (not self._normalize) or (self._asymmetric_alpha) or (self._tversky_alpha) or (self._tversky_beta) or (self._row_weights) or (self._shrink):
                print("Options normalize, asymmetric_alpha, tversky_alpha, tversky_beta, row_weights are ignored with standard implementation. Try with implementation: aiolli")
            self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)
        
    def get_single_recommendation(self, mask, k, *args):
        ic()
        ic("calling get_user_recs() k times: ".format(k))
        ic(k)
        user_recs = {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}
        
        with open("data/movielens_2k/get_user_recs.txt", "w") as f:
            for u, recs in user_recs.items():
                f.writelines(str(u) + " : " + str(recs) + "\n\n")
                
        
        return user_recs
    
    def get_user_recs_df(self, recs):
        ic()
        #from_records_list = []
        from_records_list = [(u, rec[0], rec[1]) for u, rec_list in recs.items() for rec in rec_list]

        #for u, rec_list in recs.items():
        #    for rec in rec_list:
        #        from_records_list.append((u, rec[0], rec[1]))
            #         NEW_REC = pd.DataFrame({"userId" : [75], "itemId" : [1], "rating": [5]}) # deleted ,

        df = pd.DataFrame.from_records(from_records_list, columns = ["userId", "itemId", "rating"])
        ic(df[df["rating"] > 5].sort_values(by = "rating", ascending=False).head(20))
        
        return df
        

    def get_recommendations(self, k: int = 10, **kwargs): #df = True):
        # why two?
        ic()
        ic("k in get recommendations is:")
        ic(k)
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)
        
          
        #with open("data/movielens_2k/predictions_top_k_val.txt", "w") as f:
        #    for u, recs in predictions_top_k_val.items():
        #        f.writelines(str(u) + " : " + str(recs) + "\n\n")
                
        #with open("data/movielens_2k/predictions_top_k_test.txt", "w") as f:
        #    for u, recs in predictions_top_k_test.items():
        #        f.writelines(str(u) + " : " + str(recs) + "\n\n")
                
        #ic(predictions_top_k_val == predictions_top_k_test)
        if ("plot" in kwargs) and (kwargs["plot"]):
            user_recs_df = self.get_user_recs_df(predictions_top_k_val)
            user_recs_df["rating"].plot.hist(bins = 50)
            plt.show()
                
        if ("df" in kwargs) and (kwargs["df"]):
            user_recs_df = self.get_user_recs_df(predictions_top_k_val)
            return user_recs_df
            #self.get_user_recs_df(predictions_top_k_test)
        
        return predictions_top_k_val, predictions_top_k_test

    def evaluate(self, it=None, loss=0, **kwargs):
        r"""
        what do I do?
        get recommendations and evaluate!!!
        """
        
        ic()
        if (it is None) or (not (it + 1) % self._validation_rate):
            
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations(), **kwargs)
            result_dict = self.evaluator.eval(recs)
            
            #ic(result_dict)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        self._model.save_weights(self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")


    @property
    def name(self):
        return f"UserKNN_{self.get_params_shortcut()}"

    def train(self):
        ic()

        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")
        
        prs = {"df" : False, "plot" : True}
        self.evaluate(**prs)

