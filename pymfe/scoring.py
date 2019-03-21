from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def kappa(y_true, y_pred):
    raise NotImplementedError('The "kappa" score was not implemented.')

def auc(y_true, y_pred):
    raise NotImplementedError('The "auc" score was not implemented.')
