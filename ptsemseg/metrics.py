# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

def _confusion_matrix(gt, pred, n_classes):
    mask = (gt >= 0) & (gt < n_classes)
    cm = np.bincount(
        n_classes * gt[mask].astype(int) +
        pred[mask], minlength=n_classes**2).reshape(n_classes, n_classes)
    return cm


def scores(gts, preds, n_classes):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    cm = np.zeros((n_classes, n_classes))
    for lt, lp in zip(gts, preds):
        cm += _confusion_matrix(lt.flatten(), lp.flatten(), n_classes)
    pixel_acc = np.diag(cm).sum() / cm.sum()
    acc_cls = np.diag(cm) / cm.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mean_iu = np.nanmean(iu)
    freq = cm.sum(axis=1) / cm.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_classes), iu))

    return {'Pixel Acc: \t': pixel_acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu
