# Inspired by wkentaro code
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class Metrics(object):
    def __init__(self, n_classes, class_weights=[]):
        """
        Args:
            class_weights (list): they are used to compute iiou metric
            usually measured on cityscapes dataset
        """
        self.n_classes = n_classes
        self.metrics = {'pixel_acc' : _pixel_accuracy,
                        'mean_acc' : _mean_accuracy,
                        'iou_class' : _iou_class}
        self.class_weights = class_weights
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n_classes, self.n_classes))

    def _confusion_matrix(self, gt, pred, background=False):
        """ Compute confusion matrix
        Args:
            background (bool): whether to include or not background or void (0)
                class when computing metrics. Default: False
        """
        if background:
            mask = (gt >= 0) & (gt < self.n_classes)
        else:
            mask = (gt > 0) & (gt < self.n_classes)
        self.cm = np.bincount(
            self.n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self.n_classes**2).reshape(self.n_classes,
                                                             self.n_classes)

    def compute(self, metric_name, gts, preds):
        self.reset()
        for lt, lp in zip(gts, preds):
            self.cm += _confusion_matrix(lt.flatten(),
                                         lp.flatten(),
                                         self.n_classes)
        return self.metrics()

    def _pixel_accuracy(self):
        """Pixel-wise accuracy
        """
        pixel_acc = np.diag(self.cm).sum() / self.cm.sum()
        return pixel_acc

    def _mean_accuracy(self):
        """Pixel-wise accuracy but averaged on classes
        """
        acc_cls = np.diag(self.cm) / self.cm.sum(axis=1)
        return np.nanmean(acc_cls)

    def _iou_class(self):
        """Intersection over Union averaged on classes
        """
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        return np.nanmean(iou)

    def _iiou_class(self):
        """Intersection over Union averaged on classes weighted by
           average instance size
        """
        tp = np.diag(cm) * self.class_weights
        fp = cm.sum(axis=1)
        fn = cm.sum(axis=0) * self.class_weights
        iiou = tp / (fp + fn - tp)
        return np.nanmean(iiou)
