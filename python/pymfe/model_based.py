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

    leaf = model.apply(X)
    node = model.tree_.node_count

    table = np.zeros((node, 4))
    table[:,0] = model.tree_.feature
    table[:,2] = model.tree_.n_node_samples

    tmp = np.array([leaf, y])

    for x in set(leaf):
        table[x,3] = Counter(tmp[1,tmp[0,:] == x]).keys()[0] + 1
        table[x,1] = 1

    return table

def _leaves(model, X, y):
    aux = extract(model, X, y)
    return aux.sum(axis=0)[1]

def _treeDepth(tree):

    def nodeDepth(node, depth, l, r, depths):
        depths += [depth]
        if l[node] != -1 and r[node] != -1:
            nodeDepth(l[node], depth + 1, l, r, depths)
            nodeDepth(r[node], depth + 1, l, r, depths)

    depths = []
    nodeDepth(0, 0, tree.children_left, tree.children_right, depths)
    return np.array(depths)

def _leavesBranch(model, X, y):
    return _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]

def _leavesCorrob(model, X, y):
    aux = extract(model, X, y)
    return aux[:,2][aux[:,1] == 1]/len(X)

def _treeShape(model, X, y):
    aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 1]
    return np.log2(aux)

def _leavesHomo(model, X, y):
    return _leaves(model, X, y)/_treeShape(model, X, y)

def _leavesPerClass(model, X, y):
    aux = Counter(extract(model, X, y)[:,3]).items()
    aux = aux/_leaves(model, X, y)
    return aux[1:,1]

def _nodes(model, X, y):
    return sum(extract(model, X, y)[:,1] != 1)

def _nodesPerAttr(model, X, y):
    return _nodes(model, X, y)/X.shape[1]

def _nodesPerInst(model, X, y):
    return _nodes(model, X, y)/len(X)

def _nodesPerLevel(model, X, y):
    aux = _treeDepth(model.tree_)[extract(model, X, y)[:,1] == 0]
    aux = np.array(Counter(aux).items())[:,1]
    return aux/_leaves(model, X, y)

def _nodesRepeated(model, X, y):
    aux = extract(model, X, y)[:,0] 
    aux = aux[aux > 0]
    aux = np.array(Counter(aux).items())[:,1]
    return aux

def _treeImbalance(model, X, y):
    pass

def _varImportance(model, X, y):
    return model.tree_.compute_feature_importances()

