import sys
import os.path
import torch
import visdom
import argparse
import time
import math

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import AverageMeter
from ptsemseg.metrics import MultiAverageMeter
from ptsemseg.metrics import Metrics

def main(args):
    global vis

    # Setup Dataset and Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    train_dataset = data_loader(data_path,
                         is_transform=True,
                         img_size=(args.img_rows, args.img_cols))
    args.n_classes = train_dataset.n_classes
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True)

    # Setup visdom for visualization
    vis = visdom.Visdom()

    # Setup Model
    model = get_model(args.arch, args.n_classes)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        test_image, test_segmap = train_dataset[0]
        test_image = Variable(test_image.unsqueeze(0).cuda(0))
    else:
        test_image, test_segmap = train_dataset[0]
        test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=5e-4)

    # Main training loop
    for epoch in range(args.n_epoch):
        train(trainloader, model, cross_entropy2d, optimizer, epoch, args)

        validate(valloader, model, cross_entropy2d, epoch, args)

        # Visualize result
        test_output = model(test_image)
        predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        target = loader.decode_segmap(test_segmap.numpy())
        vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

        # Save model
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model, os.path.join(args.save_path, "{}_{}_{}_{}.pkl".format(args.arch,
                                                                                args.dataset,
                                                                                args.feature_scale, epoch)))

def train(trainloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_time = AverageMeter()
    losses = AverageMeter()
    multimeter = MultiAverageMeter(len(args.metrics))
    metrics = Metrics(n_classes=args.n_classes)

    # Initialize current epoch log
    epoch_loss_window = vis.line(X=torch.zeros(1),
                           Y=torch.zeros(1),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Single Epoch Training Loss',
                                     legend=['Loss']))

    model.train()

    end = time.perf_counter()
    for i, (images, labels) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0, async=True))
            labels = Variable(labels.cuda(0, async=True))
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)

        #Compute metrics
        start_eval_time = time.perf_counter()
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        values = metrics.compute(args.metrics, gt, pred)
        multimeter.update(values, images.size(0))
        eval_time.update(time.perf_counter() - start_eval_time)

        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        vis.line(
            X=torch.ones(1) * i,
            Y=torch.Tensor([loss.data[0]]),
            win=epoch_loss_window,
            update='append')

        batch_log_str = ('Epoch: [{}/{}][{}/{}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Eval {eval_time.val:.3f} ({eval_time.avg:.3f})\t'
                        'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(
                           epoch+1, args.n_epoch, i,
                           math.floor(trainloader.dataset.__len__()/trainloader.batch_size),
                           batch_time=batch_time, data_time=data_time,
                           eval_time=eval_time, loss=losses))
        for i,m in enumerate(args.metrics):
            batch_log_str += ' {}: {:.3f} ({:.3f})'.format(m ,
                                                           multimeter.meters[i].val,
                                                           multimeter.meters[i].avg)
        print(batch_log_str)

    vis.close(win=epoch_loss_window)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--save_path', nargs='?', type=str, default='.',
                        help='Location where checkpoints are saved')
    parser.add_argument('--metrics', nargs='?', type=str, default='pixel_acc,iou_class',
                        help='Metrics to compute and show')
    parser.add_argument('--num_workers', nargs='?', type=int, default=4,
                        help='Number of processes to load and preprocess images')
    args = parser.parse_args()
    #Params preprocessing
    args.metrics = args.metrics.split(',')
    # Call main function
    main(args)
