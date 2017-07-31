import numpy as np
from sklearn.calibration import calibration_curve


def top_k_accuracy(true, pred, k=5):
    n_samples = len(true)
    hits = np.equal(pred.argsort(1)[:, -k:], true.reshape(-1, 1)).sum()
    return hits/n_samples


def per_class_accuracy(true, pred):

    true_ohehot = np.zeros((len(true), 256))
    for i in range(len(true)):
        true_ohehot[i, true[i]] = 1.0

    pred_onehot = np.equal(pred, pred.max(1).reshape(-1, 1)).astype('int')

    # 20 samples per class
    return (true_ohehot*pred_onehot).sum(0)/20.0


def entropy(pred):

    prob = pred.astype('float64')
    log = np.log2(prob)
    result = -(prob*log).sum(1)

    return result


def model_calibration(true, pred):
    pass


def count_params(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count
