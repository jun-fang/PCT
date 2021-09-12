import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.nn.functional as F

import sys
import math
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

from resnet_cifar import *

# referece: https://github.com/pytorch/examples/blob/master/imagenet/main.py

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--data-dir', default='data', type=str, metavar='PATH',
                    help='path to the dataset')
parser.add_argument('--dataset', default='imagenet', type=str, 
                    help='dataset: cifar10, cifar100, or imagenet')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                    help='warmup epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=5e-4, type=float, 
                    help='minimum learning rate in CosineAnnealingLR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


## positive-congruent training
parser.add_argument('--pct-opt', default=None, type=str, 
                    choices=[None, 'None', 'naive', 'fd-kl', 'fd-lm', 'FD-KL', 'FD-LM'],
                    help='use None, naive baseline, Focal Distillation - KL or Logit Matching '
                    ' for positive-congruent training')
parser.add_argument('--old-arch', metavar='ARCH', default=None,
                    help='model architecture (default: none)')
parser.add_argument('--old-pretrained', action='store_true',
                    help='use pre-trained old model')
parser.add_argument('--old-model-path', default=None, type=str, metavar='PATH',
                    help='path to old model path (default: none)')
parser.add_argument('--alpha', default=1.0, type=float, 
                    help='alpha')
parser.add_argument('--beta', default=5.0, type=float, 
                    help='beta')
parser.add_argument('--temperature', default=100.0, type=float, 
                    help='temperature')
parser.add_argument('--dropout-ratio', default=0.0, type=float, 
                    help='dropout ratio')


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    total_start_time = time.time()

    if '-' in str(args.pct_opt):
        args.pct_opt = args.pct_opt.upper()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    if args.dataset.lower().startswith('cifar'):
        num_classes = 100 if '100' in args.dataset else 10
        Net = ResNet18_CIFAR if '18' in args.arch else ResNet50_CIFAR
        model = Net(num_classes=num_classes, dropout_ratio=args.dropout_ratio)
        if args.old_arch is not None:
            old_model = Net(num_classes=num_classes, dropout_ratio=args.dropout_ratio)
    elif args.dataset.lower() == 'imagenet':
        num_classes = 1000 
        model = models.__dict__[args.arch](pretrained=args.pretrained)
        if args.old_arch is not None:
            old_model = models.__dict__[args.old_arch](pretrained=args.old_pretrained)
    else:
        raise ValueError('This script is only for CIFAR or ImageNet dataset!')
    if args.old_arch is None:
        old_model = None
        print('old model is None !!!')
    
    # load model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if old_model is not None:
                old_model.cuda(args.gpu)
                old_model = torch.nn.parallel.DistributedDataParallel(old_model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if old_model is not None:
                old_model.cuda()
                old_model = torch.nn.parallel.DistributedDataParallel(old_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if old_model is not None:
            old_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            if old_model is not None:
                old_model = torch.nn.DataParallel(old_model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.old_model_path is not None:
        old_checkpoint = torch.load(args.old_model_path)
        old_model.load_state_dict(old_checkpoint['state_dict'])
        print("=> loaded old checkpoint '{}' (epoch {})"
              .format(args.old_model_path, old_checkpoint['epoch']))

        for para in old_model.parameters():
            para.requires_grad = False


    cudnn.benchmark = True

    # Data loading code
    if args.dataset.lower().startswith('cifar'):
        dataset_mean = [0.491, 0.482, 0.447] if args.dataset.lower() == 'cifar10' else [0.5071, 0.4865, 0.4409]
        dataset_std = [0.247, 0.243, 0.262] if args.dataset.lower() == 'cifar10' else [0.2673, 0.2564, 0.2762]
        
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        val_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        if args.dataset.lower() == 'cifar10':
            train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=val_transform)
        else:
            train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=val_transform)
    else:
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    # evaluation
    if args.evaluate:
        if old_model is not None:
            calculate_NFR(val_loader, old_model, model, criterion, args)
        else:
            validate(val_loader, model, criterion, args)
        return


    # creat logs
    create_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i in range(100):
        exp_num = '0' + str(i + 1) if i < 10 else str(i + 1)
        folder_name = '%s-%s-PCT:%s-exp%s/' % (args.dataset, args.arch, args.pct_opt, exp_num)
        checkpoint_folder = 'checkpoints/' + folder_name
        if not os.path.exists(checkpoint_folder):
            break
    os.makedirs(checkpoint_folder, exist_ok = True)

    log = Logger()
    log.open(checkpoint_folder + 'log.train.txt', mode='a')
    log.write('\n')
    log.write(str(args))
    log.write('\n')


    # training
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, old_model=old_model)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, fileforlder=checkpoint_folder)

            epoch_lr = get_learning_rate(optimizer)
            epoch_time = time.time() - epoch_start_time

            log.write('\nEpoch %d/%d, lr: %.6f, time: %.3f min; valid Acc@1: %.3f, best Acc@1: %.3f' 
                % (epoch, args.epochs, epoch_lr, epoch_time / 60, acc1, best_acc1))
            print('\n')

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        table_path = 'checkpoints/all_results.csv'
        new_df = pd.DataFrame({
            'dataset': [args.dataset],
            'model': [args.arch], 
            'PCT': [args.pct_opt],
            'batch_size': [args.batch_size],
            'epochs': [args.epochs],
            'top1_acc': [float(best_acc1)],
            'time': [('%.2f hours' % ((time.time() - total_start_time) / 3600), create_time)],
            'model_dir': [folder_name]})
        new_df = new_df[['dataset', 'model', 'PCT', 'batch_size', 'epochs', 'top1_acc', 'time', 'model_dir']]
        if os.path.exists(table_path):
          old_df = pd.read_csv(table_path)
          new_df = old_df.append(new_df)
        new_df.to_csv(table_path, index = False)


def train(train_loader, model, criterion, optimizer, epoch, args, old_model=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if old_model is not None:
        old_model.eval()  # fix old model

        PCT_losses = AverageMeter('PCT-%s-Loss' % args.pct_opt, ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, PCT_losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cosine_learning_rate(optimizer, len(train_loader), i + 1, epoch, args)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if old_model is not None:
            old_output = old_model(images)
            old_pred = torch.argmax(old_output, dim=1)

            sample_weights = (old_pred == target).type(loss.dtype)

            if args.pct_opt.lower() == 'naive':
                log_prob = F.log_softmax(output, dim=1)
                log_loss = -log_prob.index_select(1, target).diag()
                PCT_loss = torch.mean(sample_weights * log_loss)
            elif args.pct_opt.lower() == 'fd-kl':
                KL_loss = nn.KLDivLoss(reduction='none').cuda(args.gpu)(
                    F.log_softmax(output / args.temperature, dim=1),         
                    F.softmax(old_output / args.temperature, dim=1)) * (args.temperature * args.temperature)
                # KL_loss = torch.mean(KL_loss, dim=1)
                KL_loss = torch.sum(KL_loss, dim=1)
                PCT_loss = torch.mean((args.alpha + args.beta * sample_weights) * KL_loss)
            elif args.pct_opt.lower() == 'fd-lm':
                l2_loss = torch.mean((output - old_output).pow(2), dim=1) / 2
                PCT_loss = torch.mean((args.alpha + args.beta * sample_weights) * l2_loss)
            
            loss = loss + PCT_loss
            PCT_losses.update(PCT_loss.item(), images.size(0))
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(train_loader):
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i + 1 == len(val_loader):
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def calculate_NFR(val_loader, old_model, new_model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    old_model.eval()
    new_model.eval()

    num_samples = 0
    true_positives, true_negatives = [], []
    false_positives, false_negatives = [], []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            old_output = old_model(images)
            new_output = new_model(images)
            old_loss = criterion(old_output, target)
            new_loss = criterion(new_output, target)   

            # measure accuracy and record loss
            acc1, acc5 = accuracy(new_output, target, topk=(1, 5))
            losses.update(new_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            old_output = old_output.data.cpu().numpy()
            new_output = new_output.data.cpu().numpy()

            old_output_batch = np.argmax(old_output, axis=1)
            new_output_batch = np.argmax(new_output, axis=1)
            labels_batch = target.data.cpu().numpy()

            batch_samples = labels_batch.size
            num_samples += batch_samples

            true_positives_batch = ((old_output_batch == labels_batch) * (new_output_batch == labels_batch)).astype(int)
            true_positives.append(np.reshape(true_positives_batch, (batch_samples, 1)))

            true_negatives_batch = ((old_output_batch != labels_batch) * (new_output_batch != labels_batch)).astype(int)
            true_negatives.append(np.reshape(true_negatives_batch, (batch_samples, 1)))

            false_positives_batch = ((old_output_batch != labels_batch) * (new_output_batch == labels_batch)).astype(int)
            false_positives.append(np.reshape(false_positives_batch, (batch_samples, 1)))

            false_negatives_batch = ((old_output_batch == labels_batch) * (new_output_batch != labels_batch)).astype(int)
            false_negatives.append(np.reshape(false_negatives_batch, (batch_samples, 1)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i + 1 == len(val_loader):
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    true_positives = np.vstack(true_positives)
    true_negatives = np.vstack(true_negatives)
    false_positives = np.vstack(false_positives)
    false_negatives = np.vstack(false_negatives)

    old_error_rate = (1 - (np.sum(true_positives) + np.sum(false_negatives)) / num_samples)
    new_error_rate = (1 - (np.sum(true_positives) + np.sum(false_positives)) / num_samples)

    negative_flip_rate = np.sum(false_negatives) / num_samples
    relative_negative_flip_rate = negative_flip_rate / ((1 - old_error_rate) * new_error_rate)

    print('num_samples: ', num_samples)
    print('old_error_rate: %.2f' % (old_error_rate * 100), int(old_error_rate * num_samples))
    print('new_error_rate: %.2f' % (new_error_rate * 100), int(new_error_rate * num_samples))
    print('negative_flip_rate: %.2f' % (negative_flip_rate * 100), int(negative_flip_rate * num_samples))
    print('relative_negative_flip_rate: %.2f\n' % (relative_negative_flip_rate * 100))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', fileforlder=''):
    filename = fileforlder + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, fileforlder + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_learning_rate(optimizer, iter_in_epoch, iteration, epoch, args):
    """Cosine learning rate decay with several epochs warmups"""
    init_lr, min_lr, warmup_epochs = args.lr, args.min_lr, args.warmup_epochs
    if epoch < warmup_epochs:
        warmup_iters = warmup_epochs * iter_in_epoch
        current_iter = epoch * iter_in_epoch + iteration
        lr = min_lr + current_iter / warmup_iters * (init_lr - min_lr)
    else:
        max_iters = (args.epochs - warmup_epochs) * iter_in_epoch
        current_iter = (epoch - warmup_epochs) * iter_in_epoch + iteration
        temp = (math.cos(math.pi * current_iter / max_iters) + 1) / 2
        lr = min_lr + (init_lr - min_lr) * temp
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()