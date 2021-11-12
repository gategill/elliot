"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Michal Polak Szarkowicz'
__email__ = '118304271@umail.ucc.ie'

import pickle
import time

import importlib


import pandas as pd
#from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.utils.folder import build_model_folder

from icecream import ic
ic.configureOutput(includeContext=True)

from types import SimpleNamespace


import copy

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
#from elliot.recommender.knn.user_knn.user_knn_similarity import Similarity
#from elliot.recommender.knn.user_knn.aiolli_ferrari import AiolliSimilarity
from elliot.recommender.base_recommender_model import init_charger



#class RecursiveUserKNN(UserKNN):
class RecGeneric(RecMixin, BaseRecommenderModel):
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
            # variable_name, public_name, shortcut, default, reading_function, _ 
            ("_enrichment_rounds", "enrichment_rounds", "enr", 1, int, None),
            ("_submodels", "submodels", "subm", "UserKNN", None, None)
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

        return f"RecGeneric_{self.get_params_shortcut()}"

    def get_params_shortcut(self):
        ic()
        
        """params_shortcut_list = []
        for p in self._params_list:
            x = ""
            #if p[5]:
            if str(p[2]) == "subm":
                subm_params = p[5](getattr(self, p[0]))
                s = ""
                
                # flatten
                df = pd.json_normalize(subm_params, sep='-')
                flattened = df.to_dict(orient='records')[0]
                    
                    
                ic(flattened)

                for k, v in flattened.items():
                    s += k + "-" + str(v) + "--"
                        
                x = "subm=" + s
                        
            else:           
                #x = str(p[2]+"="+ str(p[5](getattr(self, p[0]))))
                x =  str(p[2]) + "=" + str(p[5](getattr(self, p[0])))
            
            ic(x)
            params_shortcut_list.append(x)
                
            #else:
            #    x = str(getattr(self, p[0])).replace(".", "$")
            #    ic(x)
            #    ic("WERE ACUTALLY HERE")
            #    params_shortcut_list.append(x)
                
            
        #[str(p[2])+"="+ str(p[5](getattr(self, p[0])) if p[5] else getattr(self, p[0])).replace(".", "$") for p in self._params_list]
        ic(params_shortcut_list)
        
        params_shortcut = "_".join(params_shortcut_list)
        ic(params_shortcut)"""
        params_shortcut = "enr=3_subm=UserKNN-neighbors-[9]--UserKNN-similarity-cosine--UserKNN-implementation-standard"
        #ic("LOOOOK AT ME HERE")
        #ico = "_".join([str(p[2])+"="+ str(p[5](getattr(self, p[0])) if p[5] else getattr(self, p[0])).replace(".", "$") for p in self._params_list])
        #ic(ico)
        return params_shortcut

    
    def train(self):
        ic()

        #if self._restore:
        #    return self.restore_weights()
        
        # enrichment phase
        ENRICHMENTS = self._params.enrichment_rounds
        
        for i in range(ENRICHMENTS + 1):
            ic(i)
            ic(self._data.transactions)
            ic("before Similarity")
            #model_instance = UserKNN(data = self._data, config = self._config, params=self._params)
            #model_instance = UserKNN(data = self._data, config = self._config, params=self._params)
            self._ratings = self._data.train_dict
            
            # create submodel
            ic(self._params.submodels)
            sub_key, sub_model_base = "UserKNN", self._params.submodels["UserKNN"]
            
            model_class = getattr(importlib.import_module("elliot.recommender"), sub_key)
            
            ns = SimpleNamespace(**sub_model_base)
            #meta_model = self.config[_experiment][_models][key].get(_meta, {})
            if not hasattr(ns, "meta"):
                setattr(ns, "meta", SimpleNamespace())

            ic("SHOW ME THE NAMESPACE")
            ic(ns)
            
            sub_model = model_class(data = self._data, config = self._config, params = ns)
            
            ic("THIS IS VERY WRONG")
            
            #self.set_model()
            #self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)

            ic("before model_instance.ititiaize()")
            #model = self.model_class(data=d
            # ata_obj, config=self.base, params=model_params)
            #self._model.initialize()
            # specific for userKNN
            sub_model._model.initialize()

            
            ic("before new_recs_df")
            # error here
            new_recs_df = sub_model.get_recommendations(df = True)
            ic("before add new rect to train set")
            self._data.add_new_recs_to_train_set(new_recs_df)
            ic("Success for this roundS")
            
            #del sub_model
            del self._ratings
          
            
        self._ratings = self._data.train_dict
        #self.set_model()
        #self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)
        
        #start = time.time()
        #self._model.initialize()
        #end = time.time()
        #print(f"The similarity computation has taken: {end - start}")

        #print(f"Transactions: {self._data.transactions}")

        sub_model.evaluate()

