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
            table = MFEModelBased.extract_table(X, y, model)
            tree_depth = MFEModelBased.tree_depth(model)
            prepcomp_vals["model"] = model
            prepcomp_vals["table"] = table
            prepcomp_vals["tree_depth"] = tree_depth

        return prepcomp_vals

    @classmethod
    def extract_table(cls, X: np.array, y: np.array, model):

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
    def ft_leaves(cls, table):
        return sum(table[:, 1])

    @classmethod
    def ft_treeDepth(cls, tree_depth):
        return tree_depth

    @classmethod
    def tree_depth(cls, model):

        def nodeDepth(node, depth, l, r, depths):
            depths += [depth]
            if l[node] != -1 and r[node] != -1:
                nodeDepth(l[node], depth + 1, l, r, depths)
                nodeDepth(r[node], depth + 1, l, r, depths)

        depths = []
        nodeDepth(0, 0, model.tree_.children_left,
                  model.tree_.children_right, depths)
        return np.array(depths)

    @classmethod
    def ft_leavesBranch(cls, tree_depth, table):
        return tree_depth[table[:, 1] == 1]

    @classmethod
    def ft_leavesCorrob(cls, X: np.array, table):
        return table[:, 2][table[:, 1] == 1]/X.shape[0]

    @classmethod
    def ft_treeShape(cls, table, tree_depth):
        aux = tree_depth[table[:, 1] == 1]
        return -(1.0/2**aux) * np.log2(1.0/2**aux)

    @classmethod
    def ft_leavesHomo(cls, table, tree_depth):
        leaves = MFEModelBased.ft_leaves(table)
        tree_shape = MFEModelBased.ft_treeShape(table, tree_depth)
        return leaves/tree_shape

    @classmethod
    def ft_leavesPerClass(cls, table):
        aux = np.array(list(Counter(table[:,3]).values()))
        aux = aux[1:]/MFEModelBased.ft_leaves(table)
        return aux

    @classmethod
    def ft_nodes(cls, table):
        return sum(table[:, 1] != 1)

    @classmethod
    def ft_nodesPerAttr(cls, table, X: np.array):
        return MFEModelBased.ft_nodes(table)/X.shape[1]

    @classmethod
    def ft_nodesPerInst(cls, table, X: np.array):
        return MFEModelBased.ft_nodes(table)/X.shape[0]

    @classmethod
    def ft_nodesPerLevel(cls, table, tree_depth):
        aux = tree_depth[table[:, 1] == 0]
        aux = np.array(list(Counter(aux).values()))
        return aux

    @classmethod
    def ft_nodesRepeated(cls, table):
        aux = table[:, 0][table[:, 0] > 0]
        aux = np.array(list(Counter(aux).values()))
        return aux

    @classmethod
    def ft_varImportance(cls, model):
        return model.tree_.compute_feature_importances()

    def ft_treeImbalance(cls, table, tree_depth):
        aux = 1.0/2**tree_depth[table[:, 1] == 1]
        tmp = np.unique(aux, return_counts=True)
        tmp = tmp[0] * tmp[1]
        return -(1.0/2**tmp) * np.log2(1.0/2**tmp)
