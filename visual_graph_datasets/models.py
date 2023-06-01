"""
This module contains all the code which is relevant for models which want to work with the visual graph
dataset format. This primarily includes class interfaces which models can implement to signify that they
support certain VGD related operations
"""
import typing as t

import visual_graph_datasets.typing as tc


class PredictGraphMixin:
    """
    This class is essentially an interface, which can be implemented by a model class to ensure support for
    the functionality of performing predictions directly on the VGD related data structure of a
    ``GraphDict``.
    """
    def predict_graph(self,
                      graph: tc.GraphDict
                      ) -> t.Any:
        """
        This method must implement the functionality where the model is queried with the GraphDict
        representation of an element. The return type may be anything.
        """
        raise NotImplemented()

    def predict_graphs(self,
                       graph_list: t.List[tc.GraphDict],
                       ) -> t.List[t.Any]:
        """
        This method must implement the functionality where the model is queried with a list of GraphDicts
        and must return a list of corresponding predicted values. The return type may be anything.
        """
        raise NotImplemented()
