"""Module dedicated to extraction of Model Based Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import math
import numpy as np
from collections import Counter
from sklearn import tree

def dt(X, y):
    model = tree.DecisionTreeClassifier()
    return model.fit(X, y)

def extract(model, X, y):

    table = np.zeros((model.tree_.node_count, 4))
    table[:,0] = model.tree_.feature
    table[:,2] = model.tree_.n_node_samples

    leaves = model.apply(X)
    tmp = np.array([leaves, y + 1])

    for x in set(leaves):
        table[x,3] = Counter(tmp[1,tmp[0,:] == x]).keys()[0]
        table[x,1] = 1

    return table

def ft_leaves(model, X, y):
    aux = extract(model, X, y)
    return sum(aux[:,1])

def ft_treeDepth(model, X, y):

    def nodeDepth(node, depth, l, r, depths):
        depths += [depth]
        if l[node] != -1 and r[node] != -1:
            nodeDepth(l[node], depth + 1, l, r, depths)
            nodeDepth(r[node], depth + 1, l, r, depths)

    depths = []
    nodeDepth(0, 0, model.tree_.children_left, 
        model.tree_.children_right, depths)
    return np.array(depths)

def ft_leavesBranch(model, X, y):
    return _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]

def ft_leavesCorrob(model, X, y):
    aux = extract(model, X, y)
    return aux[:,2][aux[:,1] == 1]/len(X)

def ft_treeShape(model, X, y):
    aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]
    return np.log2(aux)

def ft_leavesHomo(model, X, y):
    return _leaves(model, X, y)/_treeShape(model, X, y)

def ft_leavesPerClass(model, X, y):
    aux = Counter(extract(model, X, y)[:,3]).items()
    aux = aux/_leaves(model, X, y)
    return aux[1:,1]

def ft_nodes(model, X, y):
    return sum(extract(model, X, y)[:,1] != 1)

def ft_nodesPerAttr(model, X, y):
    return _nodes(model, X, y)/X.shape[1]

def ft_nodesPerInst(model, X, y):
    return float(_nodes(model, X, y))/len(X)

def ft_nodesPerLevel(model, X, y):
    aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 0]
    return np.array(Counter(aux).items())[:,1]

def ft_nodesRepeated(model, X, y):
    aux = extract(model, X, y)[:,0] 
    aux = aux[aux > 0]
    aux = np.array(Counter(aux).items())[:,1]
    return aux

def ft_treeImbalance(model, X, y):
    pass

def ft_varImportance(model, X, y):
    return model.tree_.compute_feature_importances()

