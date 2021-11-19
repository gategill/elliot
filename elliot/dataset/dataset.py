"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import copy
import os
from types import SimpleNamespace

from icecream import ic
ic.configureOutput(includeContext=True)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typing as t
import logging as pylog


from elliot.dataset.abstract_dataset import AbstractDataset
from elliot.splitter.base_splitter import Splitter
from elliot.prefiltering.standard_prefilters import PreFilter
from elliot.negative_sampling.negative_sampling import NegativeSampler
from elliot.utils import logging

from elliot.dataset.modular_loaders.loader_coordinator_mixin import LoaderCoordinator


class DataSetLoader(LoaderCoordinator):
    """
    Load train and test dataset
    """

    def __init__(self, config, *args, **kwargs):
        ic()
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """

        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        if config.config_test:
            return
        if config.data_config.strategy == "fixed":
            path_train_data = config.data_config.train_path
            path_val_data = getattr(config.data_config, "validation_path", None)
            path_test_data = config.data_config.test_path

            self.train_dataframe = pd.read_csv(path_train_data, sep="\t", header=None, names=self.column_names)
            self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)

            # self.train_dataframe, self.side_information = self.coordinate_information(self.train_dataframe, sides=config.data_config.side_information)
            # self.train_dataframe = pd.read_csv(path_train_data, sep="\t", header=None, names=self.column_names)

            self.train_dataframe = self.check_timestamp(self.train_dataframe)
            self.test_dataframe = self.check_timestamp(self.test_dataframe)

            self.logger.info(f"{path_train_data} - Loaded")

            if config.binarize == True or all(self.train_dataframe["rating"].isna()):
                self.test_dataframe["rating"] = 1
                self.train_dataframe["rating"] = 1

            if path_val_data:
                self.validation_dataframe = pd.read_csv(path_val_data, sep="\t", header=None, names=self.column_names)
                self.validation_dataframe = self.check_timestamp(self.validation_dataframe)

                if config.binarize == True or all(self.train_dataframe["rating"].isna()):
                    self.validation_dataframe["rating"] = 1

                self.tuple_list = [([(self.train_dataframe, self.validation_dataframe)], self.test_dataframe)]
                self.tuple_list, self.side_information = self.coordinate_information(self.tuple_list,
                                                                                     sides=config.data_config.side_information,
                                                                                     logger=self.logger)
            else:
                self.tuple_list = [(self.train_dataframe, self.test_dataframe)]
                self.tuple_list, self.side_information = self.coordinate_information(self.tuple_list,
                                                                                     sides=config.data_config.side_information,
                                                                                     logger=self.logger)

        elif config.data_config.strategy == "hierarchy":
            self.tuple_list = self.read_splitting(config.data_config.root_folder, column_names=self.column_names)

            self.tuple_list, self.side_information = self.coordinate_information(self.tuple_list, sides=config.data_config.side_information,
                                                                                     logger=self.logger)

        elif config.data_config.strategy == "dataset":
            self.logger.info("There will be the splitting")
            path_dataset = config.data_config.dataset_path
            ic(path_dataset)

            self.dataframe = pd.read_csv(path_dataset, sep="\t", header=None, names=self.column_names)
            self.dataframe, self.side_information = self.coordinate_information(self.dataframe,
                                                                                sides=config.data_config.side_information,
                                                                                logger=self.logger)
            # self.dataframe = pd.read_csv(path_dataset, sep="\t", header=None, names=self.column_names)

            self.dataframe = self.check_timestamp(self.dataframe)

            self.logger.info(('{0} - Loaded'.format(path_dataset)))

            self.dataframe = PreFilter.filter(self.dataframe, self.config)

            if config.binarize == True or all(self.dataframe["rating"].isna()):
                self.dataframe["rating"] = 1

            splitter = Splitter(self.dataframe, self.config.splitting, self.config.random_seed)
            self.tuple_list = splitter.process_splitting()

        else:
            raise Exception("Strategy option not recognized")

        if isinstance(self.tuple_list[0][1], list):
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.config.evaluation.paired_ttest = False
            self.config.evaluation.wilcoxon_test = False

    def check_timestamp(self, d: pd.DataFrame) -> pd.DataFrame:
        ic()
        if all(d["timestamp"].isna()):
            d = d.drop(columns=["timestamp"]).reset_index(drop=True)
        return d

    def read_splitting(self, folder_path, column_names):
        ic()
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = pd.read_csv(os.sep.join([folder_path, test_dir, "test.tsv"]), sep="\t", names=self.column_names)
                val_dirs = [os.sep.join([folder_path, test_dir, val_dir]) for val_dir in os.listdir(os.sep.join([folder_path, test_dir])) if os.path.isdir(os.sep.join([folder_path, test_dir, val_dir]))]
                val_list = []
                for val_dir in val_dirs:
                    train_ = pd.read_csv(os.sep.join([val_dir, "train.tsv"]), sep="\t", names=self.column_names)
                    val_ = pd.read_csv(os.sep.join([val_dir, "val.tsv"]), sep="\t", names=self.column_names)
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = pd.read_csv(os.sep.join([folder_path, test_dir, "train.tsv"]), sep="\t", names=self.column_names)
                tuple_list.append((val_list, test_))

        return tuple_list

    def generate_dataobjects(self) -> t.List[object]:
        ic()
        data_list = []
        for p1, (train_val, test) in enumerate(self.tuple_list):
            # testset level
            if isinstance(train_val, list):
                ic("train val is a list VALIDATION")
                # validation level
                val_list = []
                for p2, (train, val) in enumerate(train_val):
                    self.logger.info(f"Test Fold {p1} - Validation Fold {p2}")
                    single_dataobject = DataSet(self.config, (train,val,test), self.side_information, self.args, self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                ic("train val is not a list NO VALIDATION")

                self.logger.info(f"Test Fold {p1}")
                # needs train and test
                # somehow for tuple and somehow global
                single_dataobject = DataSet(self.config, (train_val, test), self.side_information, self.args,
                                                              self.kwargs)
                data_list.append([single_dataobject])
        return data_list
    


    def generate_dataobjects_mock(self) -> t.List[object]:
        ic()
        np.random.seed(self.config.random_seed)
        _column_names = ['userId', 'itemId', 'rating']
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)
        data_list = [[DataSet(self.config, (training_set, test_set), self.args, self.kwargs)]]

        return data_list

class DataSet(AbstractDataset):
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        ic("DataSet ___init___ is called")
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if config.config_test else
                                         pylog.DEBUG)
        self.config = config
        self.args = args
        self.kwargs = kwargs
        self._data_tuple = data_tuple
        self._side_information_data = side_information_data

        # what do you do?
        if self.config.align_side_with_train == True:
            self.side_information = self.align_with_training(train=self._data_tuple[0], side_information_data=self._side_information_data)
        else:
            self.side_information = self._side_information_data

        # update these?
        self.train_dict = self.dataframe_to_dict(self._data_tuple[0])

        self.users = list(self.train_dict.keys())
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.transactions = sum(len(v) for v in self.train_dict.values())
        
        ic(self.transactions)
        
    

        sparsity = 1 - (self.transactions / (self.num_users * self.num_items))
        self.logger.info(f"Statistics\tUsers:\t{self.num_users}\tItems:\t{self.num_items}\tTransactions:\t{self.transactions}\t"
                         f"Sparsity:\t{sparsity}")

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()
        self.sp_i_train_ratings = self.build_sparse_ratings()

        # 
        if len(self._data_tuple) == 2:
            ic("data_tuple == 2")
            self.test_dict = self.build_dict(self._data_tuple[1], self.users)
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items,
                                                                           self.private_users, self.private_items,
                                                                           self.sp_i_train, None, self.test_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)
        else:
            ic("data_tuple != 2")

            self.val_dict = self.build_dict(self._data_tuple[1], self.users)
            self.test_dict = self.build_dict(self._data_tuple[2], self.users)
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items,
                                                                           self.private_users, self.private_items,
                                                                           self.sp_i_train, self.val_dict, self.test_dict)
                sp_i_val = self.to_bool_sparse(self.val_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                val_candidate_items = val_neg_samples + sp_i_val
                self.val_mask = np.where((val_candidate_items.toarray() == True), True, False)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)

        self.allunrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)
        
        #with open("data/movielens_2k/data_tuple_0.txt", "w") as f:
        #    f.write(str(self._data_tuple[0]) + "\n\n")
        #    f.write(str(type(self._data_tuple[0])) + "\n\n")
        #    
        #    f.write(str(self.users) + "\n\n")
        #    f.write(str(self.items) + "\n\n")
        #    
        #    f.write(str(self.num_users) + "\n\n")
        #    f.write(str(self.num_items) + "\n\n")
        #    
        #    f.write(str(self.transactions) + "\n\n")
        #    
        #    f.write(str(self.private_users) + "\n\n")
        #    f.write(str(self.public_users) + "\n\n")
        #    f.write(str(self.private_items) + "\n\n")
        #    f.write(str(self.public_items) + "\n\n")
        #    f.write(str(self.allunrated_mask) + "\n\n")
        #    
        #    f.write(str(self.i_train_dict) + "\n\n")
        #    f.write(str(self.sp_i_train) + "\n\n")
        #    f.write(str(self.sp_i_train_ratings) + "\n\n")
            
        #self.add_new_recs_to_train_set()
    #def get_ratings(self, new_recs_df_rating):
        #range_of_ratings = new_recs_df_rating.rating.max() - new_recs_df_rating.rating.min()
        
        
        #return 5 - (100*())
        
        
        

    def add_new_recs_to_train_set(self, new_recs_df = None, normalise = True):
        ic()
        if new_recs_df is None:
            new_recs_df = pd.DataFrame({"userId" : [75], "itemId" : [1], "rating": [5]})
            
        if normalise:
            range_of_ratings = new_recs_df.rating.max() - new_recs_df.rating.min()
            new_recs_df["rating"] = new_recs_df.apply(lambda x : 5 - np.round(5*(x["rating"]/range_of_ratings), 3), axis = 1)
            ic("normalised ratings")
            ic(new_recs_df["rating"])
                       
        new_train_df = self._data_tuple[0].append(new_recs_df, ignore_index = True)
        
        # (train, test)
        self._data_tuple = (new_train_df, self._data_tuple[1])
        
        with open("data/movielens_2k/data_tuple_0_updated_df.txt", "w") as f:
            #for u, recs in user_recs.items():
            f.write(str(new_train_df) + "\n\n")
            
        self.update_train_set()
        
        
    def update_train_set(self):
        ic()
        
        # what do you do?
        if self.config.align_side_with_train == True:
            self.side_information = self.align_with_training(train=self._data_tuple[0], side_information_data=self._side_information_data)
        else:
            self.side_information = self._side_information_data

        # update these?
        self.train_dict = self.dataframe_to_dict(self._data_tuple[0])

        self.users = list(self.train_dict.keys())
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.transactions = sum(len(v) for v in self.train_dict.values())

        sparsity = 1 - (self.transactions / (self.num_users * self.num_items))
        self.logger.info(f"Statistics\tUsers:\t{self.num_users}\tItems:\t{self.num_items}\tTransactions:\t{self.transactions}\t"
                         f"Sparsity:\t{sparsity}")

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()
        self.sp_i_train_ratings = self.build_sparse_ratings()
        
        self.allunrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)
        ic(self.transactions)


        
        #with open("data/movielens_2k/data_tuple_0_updated.txt", "w") as f:
        #    #for u, recs in user_recs.items():
        #    f.write(str(self._data_tuple[0]) + "\n\n") # differs
        #    f.write(str(type(self._data_tuple[0])) + "\n\n")
        #    
        #    f.write(str(self.users) + "\n\n")
        #    f.write(str(self.items) + "\n\n")
        #    
        #    f.write(str(self.num_users) + "\n\n")
        #    f.write(str(self.num_items) + "\n\n")
        #    
        #    f.write(str(self.transactions) + "\n\n") # differs
        #    
        #    f.write(str(self.private_users) + "\n\n")
        #    f.write(str(self.public_users) + "\n\n")
        #    f.write(str(self.private_items) + "\n\n")
        #    f.write(str(self.public_items) + "\n\n")
        #    f.write(str(self.allunrated_mask) + "\n\n")
        #    
        #    f.write(str(self.i_train_dict) + "\n\n") # differs
        #    f.write(str(self.sp_i_train) + "\n\n") # differs
        #    f.write(str(self.sp_i_train_ratings) + "\n\n") # differs
        
    def dataframe_to_dict(self, data):
        ic()
        ratings = {k: f.groupby('itemId')['rating'].apply(float).to_dict() for k, f in data.groupby('userId')}

        """users = list(data['userId'].unique())

        "Conversion to Dictionary"
        ratings = {}
        for u in users:
            sel_ = data[data['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))"""
        return ratings

    def build_dict(self, dataframe, users):
        ic()
        ratings = {k: f.groupby('itemId')['rating'].apply(float).to_dict() for k, f in dataframe.groupby('userId') if k in users}

        """ratings = {}
        for u in users:
            sel_ = dataframe[dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))"""
        return ratings

    def build_sparse(self):
        ic()

        rows_cols = [(u, i) for u, items in self.i_train_dict.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))
        return data

    def build_sparse_ratings(self):
        ic()
        rows_cols_ratings = [(u, i, r) for u, items in self.i_train_dict.items() for i, r in items.items()]
        rows = [u for u, _, _ in rows_cols_ratings]
        cols = [i for _, i, _ in rows_cols_ratings]
        ratings = [r for _, _, r in rows_cols_ratings]

        data = sp.csr_matrix((ratings, (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))

        return data

    def get_test(self):
        ic()
        return self.test_dict

    def get_validation(self):
        ic()
        return self.val_dict if hasattr(self, 'val_dict') else None

    def to_bool_sparse(self, test_dict):
        ic()
        i_test = [(self.public_users[user], self.public_items[i])
                  for user, items in test_dict.items() if user in self.public_users.keys()
                  for i in items.keys() if i in self.public_items.keys()]
        rows = [u for u, _ in i_test]
        cols = [i for _, i in i_test]
        i_test = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='bool',
                               shape=(len(self.public_users.keys()), len(self.public_items.keys())))
        return i_test

    # what do you do?
    def align_with_training(self, train, side_information_data):
        ic()
        """Alignment with training"""
        def equal(a, b, c):
            return len(a) == len(b) == len(c)
        train_dict = self.dataframe_to_dict(train)
        users = set(train_dict.keys())
        items = set({k for a in train_dict.values() for k in a.keys()})
        users_items = []
        side_objs = []
        for k, v in side_information_data.__dict__.items():
            new_v = copy.deepcopy(v)
            users_items.append(new_v.object.get_mapped())
            side_objs.append(new_v)
        while True:
            condition = True
            new_users = users
            new_items = items
            for us_, is_ in users_items:
                temp_users = new_users & us_
                temp_items = new_items & is_
                condition &= equal(new_users, us_, temp_users)
                condition &= equal(new_items, is_, temp_items)
                new_users = temp_users
                new_items = temp_items
            if condition:
                break
            else:
                users = new_users
                items = new_items
                new_users_items = []
                for v in side_objs:
                    v.object.filter(users, items)
                    new_users_items.append(v.object.get_mapped())
                users_items = new_users_items
        ns = SimpleNamespace()
        for side_obj in side_objs:
            side_ns = side_obj.object.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)
        return ns
    
    def __str__(self):
        return "private_users: {}\n\npublic_users: {}\n\nprivate_items: {}\n\npublic_items: {}\n\nallunrated_mask: {}".format(self.private_users, self.public_users, self.private_items, self.public_items, self.allunrated_mask)
         
       
