import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


"""Tools for diagnostic if a learned system.

Arguments:
    
    true: a numpy array of shape (n_samples,) of type int
        with integers in range 0..(n_classes - 1).
    
    pred: a numpy array of shape (n_samples, n_classes) of type float,
        represents probabilities.
        
    decode: a dict that maps a class index to human readable format.
"""


def top_k_accuracy(true, pred, k=[2, 3, 4, 5]):
    n_samples = len(true)
    hits = []
    for i in k:
        hits += [np.equal(pred.argsort(1)[:, -i:], true.reshape(-1, 1)).sum()/n_samples]
    return hits


def per_class_accuracy(true, pred):

    # there are 256 classes
    true_ohehot = np.zeros((len(true), 256))
    for i in range(len(true)):
        true_ohehot[i, true[i]] = 1.0

    pred_onehot = np.equal(pred, pred.max(1).reshape(-1, 1)).astype('int')

    # 20 samples per class in the validation dataset
    per_class_acc = (true_ohehot*pred_onehot).sum(0)/20.0
    return per_class_acc


def most_inaccurate_k_classes(per_class_acc, k, decode):
    most = per_class_acc.argsort()[:k]
    for i in most:
        print(decode[i], per_class_acc[i])
    

def entropy(pred):

    prob = pred.astype('float64')
    log = np.log2(prob)
    result = -(prob*log).sum(1)

    return result


def model_calibration(true, pred, n_bins=10):
    """
    On Calibration of Modern Neural Networks,
    https://arxiv.org/abs/1706.04599
    """
    
    hits = np.equal(pred.argmax(1), true)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        hits, pred.max(1), n_bins=n_bins
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, '-ok');
    plt.plot([0.0, 1.0], [0.0, 1.0], '--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.0]);
    plt.xlabel('confidence');
    plt.ylabel('accuracy');
    plt.title('reliability curve');


def count_params(model):
    # model - pytorch's nn.Module object
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count


def most_confused_classes(val_true, val_pred, decode, min_n_confusions):
    
    conf_mat = confusion_matrix(val_true, val_pred.argmax(1))
    
    # not interested in correct predictions
    conf_mat -= np.diag(conf_mat.diagonal())
    
    # confusion(class A -> class B) + confusion(class B -> class A)
    conf_mat += conf_mat.T
    
    confused_pairs = np.where(np.triu(conf_mat) >= min_n_confusions)
    confused_pairs = [(k, confused_pairs[1][i]) for i, k in enumerate(confused_pairs[0])]
    confused_pairs = [(decode[i], decode[j]) for i, j in confused_pairs]
    
    return confused_pairs


def show_errors(erroneous_samples, erroneous_predictions, erroneous_targets, decode):
    
    n_errors = len(erroneous_targets)
    # choose 30 random erroneous predictions
    to_show = np.random.choice(np.arange(0, n_errors), size=30, replace=False)
    pictures = erroneous_samples[to_show]
    # choose top5 predictions
    pictures_predictions = erroneous_predictions.argsort(1)[:, -5:][to_show]
    pictures_probs = np.sort(erroneous_predictions, 1)[:, -5:][to_show]
    pictures_true = erroneous_targets[to_show]
    
    # values for pretrained pytorch models
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    
    pictures = np.transpose(pictures, axes=(0, 2, 3, 1))
    # reverse normalization
    pictures *= std
    pictures += mean
    
    # show pictures, predicted classes, probabilities, and true classes
    _, axes = plt.subplots(nrows=6, ncols=5, figsize=(14, 19))
    axes = axes.flatten()
    for i, pic in enumerate(pictures):
        axes[i].set_axis_off();
        axes[i].imshow(pic);

        title = decode[pictures_predictions[i][-1]] + ' ' + str(pictures_probs[i][-1]) + '\n' +\
            decode[pictures_predictions[i][-2]] + ' ' + str(pictures_probs[i][-2]) + '\n' +\
            decode[pictures_predictions[i][-3]] + ' ' + str(pictures_probs[i][-3]) + '\n' +\
            decode[pictures_predictions[i][-4]] + ' ' + str(pictures_probs[i][-4]) + '\n' +\
            decode[pictures_predictions[i][-5]] + ' ' + str(pictures_probs[i][-5]) + '\n' +\
            'true: ' + decode[pictures_true[i]]

        axes[i].set_title(title);
    
    plt.tight_layout()
