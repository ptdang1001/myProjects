from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error


def get_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return(acc)


def get_precision(y_true, y_pred):
    pc = metrics.precision_score(y_true, y_pred, average='macro')
    return(pc)


def get_sensitivity(y_true, y_pred):
    rc = metrics.recall_score(y_true, y_pred, average='macro')
    return(rc)


def get_F1(y_true, y_pred):
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return(f1)


def get_JS(y_true, y_pred):
    js = jaccard_score(y_true, y_pred, average='macro')
    return(js)


def get_RMSE(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return(rmse)
