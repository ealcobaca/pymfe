"""Module dedicated to extraction of Model Based Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import typing as t

import math
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class MFEModelBased:

    @classmethod
    def precompute_model_based_class(cls, X: np.ndarray, y: np.ndarray,
                                     random_state: t.Optional[int],
                                     **kwargs) -> t.Dict[str, t.Any]:
        """
        To do the doc string
        """
        prepcomp_vals = {}  # type: t.Dict[str, t.Any]

        if X is not None and y is not None\
           and not {"model", "table"}.issubset(kwargs):
            model = DecisionTreeClassifier(random_state=random_state)
            model.fit(X, y)
            table = MFEModelBased.extract_table(X, y, model)
            tree_depth = MFEModelBased.tree_depth(model)
            prepcomp_vals["model"] = model
            prepcomp_vals["table"] = table
            prepcomp_vals["tree_depth"] = tree_depth

        return prepcomp_vals

    @classmethod
    def extract_table(cls, X: np.ndarray, y: np.ndarray, model:
                      DecisionTreeClassifier) -> np.ndarray:

        table = np.zeros((model.tree_.node_count, 4))  # type: np.ndarray
        table[:, 0] = model.tree_.feature
        table[:, 2] = model.tree_.n_node_samples

        leaves = model.apply(X)  # type: DecisionTreeClassifier
        tmp = np.array([leaves, y + 1])  # type: np.ndarray

        x = 0  # type: int
        for x in set(leaves):
            table[x, 3] = list(Counter(tmp[1, tmp[0, :] == x]).keys())[0] + 1
            table[x, 1] = 1

        return table

    @classmethod
    def ft_leaves(cls, table: np.ndarray) -> int:
        return np.sum(table[:, 1], dtype=int)

    @classmethod
    def ft_tree_depth(cls, tree_depth: np.ndarray) -> np.ndarray:
        return tree_depth

    @classmethod
    def tree_depth(cls, model: DecisionTreeClassifier) -> np.ndarray:

        def node_depth(node: int, depth: int, l, r, depths: t.List[int]):
            depths += [depth]
            if l[node] != -1 and r[node] != -1:
                node_depth(l[node], depth + 1, l, r, depths)
                node_depth(r[node], depth + 1, l, r, depths)

        depths = []  # type: t.List[int]
        node_depth(0, 0, model.tree_.children_left,
                   model.tree_.children_right, depths)
        return np.array(depths)

    @classmethod
    def ft_leaves_branch(cls, tree_depth: np.ndarray,
                         table: np.ndarray) -> np.ndarray:
        return tree_depth[table[:, 1] == 1]

    @classmethod
    def ft_leaves_corrob(cls, X: np.ndarray, table: np.ndarray) -> np.ndarray:
        return table[:, 2][table[:, 1] == 1]/X.shape[0]

    @classmethod
    def ft_tree_shape(cls, table: np.ndarray,
                      tree_depth: np.ndarray) -> np.ndarray:
        aux = tree_depth[table[:, 1] == 1]  # type: np.ndarray
        return -(1.0/2**aux) * np.log2(1.0/2**aux)

    @classmethod
    def ft_leaves_homo(cls, table: np.ndarray,
                       tree_depth: np.ndarray) -> np.ndarray:
        leaves = MFEModelBased.ft_leaves(table)  # type: int
        tree_shape = MFEModelBased.ft_tree_shape(table,
                                                 tree_depth)  # type: np.ndarray
        return leaves/tree_shape

    @classmethod
    def ft_leaves_per_class(cls, table: np.ndarray) -> np.ndarray:
        aux = np.array(list(Counter(table[:, 3]).values()))  # np.ndarray
        aux = aux[1:]/MFEModelBased.ft_leaves(table)
        return aux

    @classmethod
    def ft_nodes(cls, table: np.ndarray) -> int:
        return np.sum(table[:, 1] != 1)

    @classmethod
    def ft_nodes_per_attr(cls, table: np.ndarray, X: np.ndarray) -> float:
        nodes = MFEModelBased.ft_nodes(table)  # type: int
        attr = X.shape[1]  # type: float
        return nodes/attr

    @classmethod
    def ft_nodes_per_inst(cls, table: np.ndarray, X: np.ndarray) -> float:
        nodes = MFEModelBased.ft_nodes(table)  # type: int 
        inst = X.shape[0]  # type: float
        return nodes/inst

    @classmethod
    def ft_nodes_per_level(cls, table: np.ndarray,
                           tree_depth: np.ndarray) -> float:
        aux = tree_depth[table[:, 1] == 0] # type: np.ndarray
        aux = np.array(list(Counter(aux).values()))
        return aux

    @classmethod
    def ft_nodes_repeated(cls, table: np.ndarray) -> np.ndarray:
        aux = table[:, 0][table[:, 0] > 0] # type: np.ndarray
        aux = np.array(list(Counter(aux).values()))
        return aux

    @classmethod
    def ft_var_importance(cls, model: DecisionTreeClassifier) -> np.ndarray:
        importance = model.tree_.compute_feature_importances()  # np.ndarray 
        return importance

    @classmethod
    def ft_tree_imbalance(cls, table: np.ndarray,
                          tree_depth: np.ndarray) -> np.ndarray:
        aux = 1.0/2**tree_depth[table[:, 1] == 1]  # np.ndarray
        tmp = np.unique(aux, return_counts=True)  # np.ndarray
        tmp = tmp[0] * tmp[1]
        return -(1.0/2**tmp) * np.log2(1.0/2**tmp)
