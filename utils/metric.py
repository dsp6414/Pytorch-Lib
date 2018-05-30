import os
import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import contextlib
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable

import matplotlib.pyplot as plt

#from sklearn import metrics as scipy_metrics



is_scalar = lambda t: torch.is_tensor(t) and len(t.size()) == 0

def torch_equals_ignore_index(tensor, tensor_other, ignore_index=None):
    """
    Compute ``torch.equal`` with the optional mask parameter.

    Args:
        ignore_index (int, optional): Specifies a ``tensor`` index that is ignored.

    Returns:
        (bool) Returns ``True`` if target and prediction are equal.
    """
    if ignore_index is not None:
        assert tensor.size() == tensor_other.size()
        mask_arr = tensor.ne(ignore_index)
        tensor = tensor.masked_select(mask_arr)
        tensor_other = tensor_other.masked_select(mask_arr)

    return torch.equal(tensor, tensor_other)

def get_accuracy(targets, outputs, k=1, ignore_index=None):
    """ Get the accuracy top-k accuracy between two tensors.

    Args:
      targets (1 - 2D :class:`torch.Tensor`): Target or true vector against which to measure
          saccuracy
      outputs (1 - 3D :class:`torch.Tensor`): Prediction or output vector
      ignore_index (int, optional): Specifies a target index that is ignored

    Returns:
      :class:`tuple` consisting of accuracy (:class:`float`), number correct (:class:`int`) and
      total (:class:`int`)

    Example:

        >>> import torch
        >>> from torchnlp.metrics import get_accuracy
        >>> targets = torch.LongTensor([1, 2, 3, 4, 5])
        >>> outputs = torch.LongTensor([1, 2, 2, 3, 5])
        >>> accuracy, n_correct, n_total = get_accuracy(targets, outputs, ignore_index=3)
        >>> accuracy
        0.8
        >>> n_correct
        4
        >>> n_total
        5
    """
    n_correct = 0.0
    for target, output in zip(targets, outputs):
        if not torch.is_tensor(target) or is_scalar(target):
            target = torch.LongTensor([target])

        if not torch.is_tensor(output) or is_scalar(output):
            output = torch.LongTensor([[output]])

        predictions = output.topk(k=min(k, len(output)), dim=0)[0]
        for prediction in predictions:
            if torch_equals_ignore_index(target, prediction, ignore_index=ignore_index):
                n_correct += 1
                break

    return n_correct / len(targets), n_correct, len(targets)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).data.cpu().numpy()[0])
    return res


def softmax_correct(preds, targets):
    pred = to_data(preds).max(1)[1] # get the index of the max log-probability
    targ = to_data(targets)
    return pred.eq(targ).cpu().type(torch.FloatTensor)

def softmax_accuracy(preds, targets, size_average=True):
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(softmax_correct(preds, targets))


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

class Meter():
    def update(self):
        raise NotImplementedError


class AverageMeter(Meter):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class MovingAverageMeter(Meter):
    def __init__(self):
        self.average = None

    def update(self, value, weight=0.1):
        if self.average is None:
            self.average = value
        else:
            self.average = (1 - weight) * self.average + weight * value


class AccuracyMeter(Meter):
    def __init__(self):
        self.correct = 0
        self.count = 0
        self.accuracy = None

    def update(self, correct, count):
        self.correct += correct
        self.count += count
        self.accuracy = self.correct / self.count



def f1_score(y_true, y_pred, sequence_lengths):
    """Evaluates f1 score.
    Args:
        y_true (list): true labels.
        y_pred (list): predicted labels.
        sequence_lengths (list): sequence lengths.
    Returns:
        float: f1 score.
    Example:
        >>> y_true = []
        >>> y_pred = []
        >>> sequence_lengths = []
        >>> print(f1_score(y_true, y_pred, sequence_lengths))
        0.8
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = set(get_entities(lab))
        lab_pred_chunks = set(get_entities(lab_pred))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return f1

def get_metric_in_blocks(outputs, targets, block_size, metric):
    sum_ = 0
    n = 0
    i = 0
    while i < len(outputs):
        out_block = outputs[i:i+block_size]
        tar_block = targets[i:i+block_size]
        score = metric(out_block, tar_block)
        sum_ += len(out_block) * score
        n += len(out_block)
        i += block_size
    return sum_ / n


def get_metrics_in_batches(model, loader, thresholds, metrics):
    model.eval()
    n_batches = len(loader)
    metric_totals = [0 for m in metrics]

    for data in loader:
        if len(data[1].size()) == 1:
            targets = data[1].float().view(-1, 1)
        inputs = Variable(data[0].cuda(async=True))
        targets = Variable(data[1].cuda(async=True))

        output = model(inputs)

        labels = targets.data.cpu().numpy()
        probs = output.data.cpu().numpy()
        preds = predictions.get_predictions(probs, thresholds)

        for i,m in enumerate(metrics):
            score = m(preds, labels)
            metric_totals[i] += score

    metric_totals = [m / n_batches for m in metric_totals]
    return metric_totals


#def get_accuracy(preds, targets):
#    preds = preds.flatten() 
#    targets = targets.flatten()
#    correct = np.sum(preds==targets)
#    return correct / len(targets)


def get_cross_entropy_loss(probs, targets):
    return F.binary_cross_entropy(
              Variable(torch.from_numpy(probs)),
              Variable(torch.from_numpy(targets).float())).data[0]


def get_recall(preds, targets):
    return scipy_metrics.recall_score(targets.flatten(), preds.flatten())


def get_precision(preds, targets):
    return scipy_metrics.precision_score(targets.flatten(), preds.flatten())


def get_roc_score(probs, targets):
    return scipy_metrics.roc_auc_score(targets.flatten(), probs.flatten())


def get_dice_score(preds, targets):
    eps = 1e-7
    batch_size = preds.shape[0]
    preds = preds.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)

    total = preds.sum(1) + targets.sum(1) + eps
    intersection = (preds * targets).astype(float)
    score = 2. * intersection.sum(1) / total
    return np.mean(score)


def get_f2_score(y_pred, y_true, average='samples'):
    y_pred, y_true, = np.array(y_pred), np.array(y_true)
    return fbeta_score(y_true, y_pred, beta=2, average=average) 


def find_f2score_threshold(probs, targets, average='samples',
                           try_all=True, verbose=False, step=.01):
    best = 0
    best_score = -1
    totry = np.arange(0.1, 0.9, step)
    for t in totry:
        score = get_f2_score(probs, targets, t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5),
              ' @ threshold =', round(best,4))
    return round(best,6)


