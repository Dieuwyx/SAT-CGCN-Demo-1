#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 创建人:王逸轩
# 日期:2025/3/6 14:45
# 软件: PyCharm

import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# 导入读取数据集、数据加载器、模型
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

warnings.filterwarnings("ignore")

# 参数解析
# 数据集路径
data_options = './dataset'
task = 'regression'
workers = 0
epochs = 100
# 起始轮数
start_epoch = 0
# 批次大小
batch_size = 256
# 学习率 0.01
lr = 1e-3
# 学习率衰减点
lr_milestones = [1e-3]
# 动量
momentum = 0.9
# 权重衰减
weight_decay = 1e-4
# 打印频率
print_freq = 10
######################################################################################
# 训练数据比例
train_ratio = 0.7
# 验证数据比例
val_ratio = 0.2
# 测试数据比例
test_ratio = 0.1
########################################################################################
# 优化器
optim_c = 'Adam'
# 原子特征长度  number of hidden atom features in conv layers
atom_fea_len = 64
# 隐藏特征长度  number of hidden features after pooling
h_fea_len = 128
# 卷积层数
n_conv = 3
# 隐藏层数  number of hidden layers after pooling
n_h = 1
# 是否启用CUDA
cuda = torch.cuda.is_available()
print("CUDA", torch.cuda.is_available())

# 最佳MAE误差
if task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


# 主函数
def main():
    # 加载数据集，输出：元组 ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    dataset = CIFData(data_options)
    # 将数据列表整理成用于预测晶体的批处理性能，打包(atom_fea, nbr_fea, nbr_fea_idx, target)
    collate_fn = collate_pool
    # 数据集划分函数
    # 输入完整数据集 + 划分参数
    # 输出：返回 train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        train_ratio=train_ratio,
        num_workers=workers,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pin_memory=cuda,
        return_test=True)
    print(len(dataset))
    # 任务类型为回归
    if len(dataset) < 500:
        warnings.warn('数据集较小，预期较低的准确性。 ')
        # 从数据集中随机选择500个样本
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        # 从数据集中随机选择500个样本
        sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]

    # 从随机取样的500样本数据列表中提取数据、目标、晶体ID
    _, sample_target, _ = collate_pool(sample_data_list)
    # 用于目标值的正则化
    normalizer = Normalizer(sample_target)

    # 构建模型
    # dataset[0]为(atom_fea, nbr_fea, nbr_fea_idx)，也就是structures
    structures, _, _ = dataset[0]
    # 原始atom_fea
    orig_atom_fea_len = structures[0].shape[-1]
    # 邻居特征长度nbr_fea
    nbr_fea_len = structures[1].shape[-1]

    print('orig_atom_fea_len',orig_atom_fea_len)
    print('nbr_fea_len',nbr_fea_len)
    print('atom_fea_len',atom_fea_len)
    '''
    orig_atom_fea_len 92
    nbr_fea_len 41
    atom_fea_len 64
    '''
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=atom_fea_len,
                                # 神经网络的超参数
                                n_conv=n_conv,
                                h_fea_len=h_fea_len,
                                n_h=n_h)
    if cuda:
        model.cuda()

    criterion = nn.MSELoss()
    if optim_c == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_c == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')


    scheduler = MultiStepLR(optimizer, milestones=lr_milestones,gamma=0.1)

    for epoch in range(start_epoch, epochs):
        # train for one epoch
        print("开始训练！")
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # 用验证
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_error and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # input为
        input_var = (Variable(input[0].cuda(non_blocking=True)),
                     Variable(input[1].cuda(non_blocking=True)),
                        input[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed.cuda(non_blocking=True))

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])

        target_normed = normalizer.norm(target)

        with torch.no_grad():
            target_var = Variable(target_normed.cuda(non_blocking=True))

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
