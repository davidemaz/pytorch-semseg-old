import sys
import torch
import visdom
import argparse
import time
import numpy as np
import math

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.loss import cross_entropy2d
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import AverageMeter
from ptsemseg.metrics import MultiAverageMeter
from ptsemseg.metrics import Metrics

def validate(valloader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_time = AverageMeter()
    losses = AverageMeter()
    multimeter = MultiAverageMeter(len(args.metrics))
    metrics = Metrics(n_classes=args.n_classes)
    model.eval()
    if torch.cuda.is_available() and not isinstance(model, nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    gts, preds = [], []
    end = time.perf_counter()
    for i, (images, labels) in enumerate(valloader):
        if args.max_iters_per_epoch != 0:
            if i > args.max_iters_per_epoch:
                break
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)
        start_eval_time = time.perf_counter()
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        values = metrics.compute(args.metrics, gt, pred)
        multimeter.update(values, images.size(0))
        eval_time.update(time.perf_counter() - start_eval_time)

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        #measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        batch_log_str = ('Val: [{}/{}][{}/{}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Eval {eval_time.val:.3f} ({eval_time.avg:.3f})\t'
                        'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(
                           epoch+1, args.n_epoch, i,
                           math.floor(valloader.dataset.__len__()/valloader.batch_size),
                           batch_time=batch_time, data_time=data_time,
                           eval_time=eval_time, loss=losses))
        for i,m in enumerate(args.metrics):
            batch_log_str += ' {}: {:.3f} ({:.3f})'.format(m ,
                                                           multimeter.meters[i].val,
                                                           multimeter.meters[i].avg)
        print(batch_log_str)


    globalValues = metrics.compute(args.metrics, gts, preds)
    print('Global Metrics:')
    for m,v in zip(args.metrics, globalValues):
        print('{}: {}'.format(m, v))
    return losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='Split of dataset to test on')
    parser.add_argument('--metrics', nargs='?', type=str, default='pixel_acc,iou_class',
                        help='Metrics to compute and show')
    parser.add_argument('--max_iters_per_epoch', nargs='?', type=int, default=0,
                        help='Max number of iterations per epoch.'
                             ' Useful for debug purposes')
    args = parser.parse_args()
    #Params preprocessing
    args.metrics = args.metrics.split(',')
    args.n_epoch = 1
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    args.n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    # Setup Model
    model = torch.load(args.model_path)
    cudnn.benchmark = True

    validate(valloader, model, cross_entropy2d, 0, args)
