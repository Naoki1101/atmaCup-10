import numpy as np
from sklearn import metrics


def rmse(y_true: np.array, y_pred: np.array) -> float:
    score = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return score


def rmsle(y_true: np.array, y_pred: np.array) -> float:
    score = np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))
    return score
