"""Module dedicated to extraction of Model Based Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import math
import numpy as np
from sklearn import tree

def dt(X, y):
    model = tree.DecisionTreeClassifier()
    return model.fit(X, Y)

def treeDepth(tree):

    def nodeDepth(c_node, c_depth, l, r, depths):
        depths += [c_depth]
        if l[c_node] != -1 and r[c_node] != -1:
            nodeDepth(l[c_node], c_depth + 1, l, r, depths)
            nodeDepth(r[c_node], c_depth + 1, l, r, depths)

    depths = []
    nodeDepth(0, 0, tree.tree_.children_left, tree.tree_.children_right, depths)
    return np.array(depths)

def extract(tree, X, y):

    leaf = tree.apply(X)
    traverse = tree.decision_path(X).todense()
    nodes = traverse.shape[1]

    table = np.zeros((nodes, 4))
    table[:,0] = range(nodes)

    table[:,2] = tree.tree_.n_node_samples

    tmp = np.array([leaf, y])

    for x in set(leaf):
        table[x,3] = Counter(tmp[1,tmp[0,:] == x]).items()[0][0] + 1
        table[x,1] = 1

    return table

def _leaves(tree, X):
    aux = extract(tree, X)
    return aux.sum(axis=0)[1]

def _leavesBranch(tree, X):
    return treeDepth(tree)[extract(tree, X)[:,1] == 1]

def _leavesCorrob(tree, X):
    aux = extract(tree, X)
    return aux[:,2][aux[:,1] == 1]/len(X)

def _treeShape(tree, X):
    aux = treeDepth(tree)[extract(tree, X)[:,1] == 1]
    return -(1 / 2 ^ aux) * np.log2(1 / 2 ^ aux)

def _leavesHomo(tree, X):
    return _leaves(tree, X)/_treeShape(tree, X)

def _leavesPerClass(tree, X):
    pass

def _nodes(tree, X):
    sum(extract(tree, X)[:,1] != 1)
