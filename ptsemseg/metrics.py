# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

def _confusion_matrix(gt, pred, n_classes):
    mask = (gt >= 0) & (gt < n_classes)
    hist = np.bincount(
        n_classes * gt[mask].astype(int) +
        pred[mask], minlength=n_classes**2).reshape(n_classes, n_classes)
    return hist


def scores(gts, preds, n_classes):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_classes, n_classes))
    for lt, lp in zip(gts, preds):
        hist += _confusion_matrix(lt.flatten(), lp.flatten(), n_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_classes), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu
