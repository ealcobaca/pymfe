"""Module dedicated to extraction of Model Based Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import typing as t

import math
import numpy as np
from collections import Counter
from sklearn import tree


class MFEModelBased:

    @classmethod
    def precompute_model_based_class(cls, X: np.ndarray, y: np.ndarray,
                                     **kwargs) -> t.Dict[str, t.Any]:
        """
        To do the doc string
        """
        prepcomp_vals = {}

        if X is not None and y is not None\
           and not {"model", "table"}.issubset(kwargs):
            model = tree.DecisionTreeClassifier()
            model.fit(X, y)
            table = MFEModelBased.extract_table(model, X, y)
            prepcomp_vals["model"] = model
            prepcomp_vals["table"] = table

        return prepcomp_vals

    # def dt(X, y):
    #     model = tree.DecisionTreeClassifier()
    #     return model.fit(X, y)

    @classmethod
    def extract_table(cls, model, X, y):

        table = np.zeros((model.tree_.node_count, 4))
        table[:, 0] = model.tree_.feature
        table[:, 2] = model.tree_.n_node_samples

        leaves = model.apply(X)
        tmp = np.array([leaves, y + 1])

        for x in set(leaves):
            table[x, 3] = list(Counter(tmp[1,tmp[0,:] == x]).keys())[0] + 1
            table[x, 1] = 1

        return table

    @classmethod
    def ft_leaves(cls, X, y, table):
        aux = table
        return sum(aux[:, 1])

    # def ft_treeDepth(X, y, model):
    #     def nodeDepth(node, depth, l, r, depths):
    #         depths += [depth]
    #         if l[node] != -1 and r[node] != -1:
    #             nodeDepth(l[node], depth + 1, l, r, depths)
    #             nodeDepth(r[node], depth + 1, l, r, depths)
    #
    #     depths = []
    #     nodeDepth(0, 0, model.tree_.children_left,
    #               model.tree_.children_right, depths)
    #     return np.array(depths)
    #
    # def ft_leavesBranch(model, X, y):
    #     return _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]
    #
    @classmethod
    def ft_leavesCorrob(cls, X, table):
        aux = table
        return aux[:, 2][aux[:, 1] == 1]/X.shape[0]

    # def ft_treeShape(model, X, y):
    #     aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]
    #     return np.log2(aux)
    #
    # def ft_leavesHomo(model, X, y):
    #     return _leaves(model, X, y)/_treeShape(model, X, y)
    #
    # def ft_leavesPerClass(model, X, y):
    #     aux = Counter(extract(model, X, y)[:,3]).items()
    #     aux = aux/_leaves(model, X, y)
    #     return aux[1:,1]
    #

    @classmethod
    def ft_nodes(cls, table):
        return sum(table[:, 1] != 1)

    @classmethod
    def ft_nodesPerAttr(cls, X, y, table):
        return MFEModelBased.ft_nodes(table)/X.shape[1]

    @classmethod
    def ft_nodesPerInst(cls, table, X):
        return float(MFEModelBased.ft_nodes(table))/X.shape[0]

    # def ft_nodesPerLevel(model, X, y):
    #     aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 0]
    #     return np.array(Counter(aux).items())[:,1]
    #
    # def ft_nodesRepeated(model, X, y):
    #     aux = extract(model, X, y)[:,0] 
    #     aux = aux[aux > 0]
    #     aux = np.array(Counter(aux).items())[:,1]
    #     return aux
    #
    # def ft_treeImbalance(model, X, y):
    #     pass
    #
    @classmethod
    def ft_varImportance(cls, model):
        return model.tree_.compute_feature_importances()

