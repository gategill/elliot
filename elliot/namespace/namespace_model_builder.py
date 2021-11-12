"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from abc import ABC, abstractmethod, abstractproperty

from elliot.namespace.namespace_model import NameSpaceModel

from icecream import ic
ic.configureOutput(includeContext=True)

class Builder(ABC):
    ic()
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @abstractproperty
    def base(self) -> None:
        pass

    @abstractmethod
    def models(self) -> None:
        pass


class NameSpaceBuilder(Builder):

    def __init__(self, config_path, base_folder_path_elliot, base_folder_path_config) -> None:
        ic()
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        self._namespace = NameSpaceModel(config_path, base_folder_path_elliot, base_folder_path_config)

    @property
    def base(self) -> NameSpaceModel:
        ic()
        namespace = self._namespace
        namespace.fill_base()
        return namespace

    def models(self) -> tuple:
        ic()
        return self._namespace.fill_model()
