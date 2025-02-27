
'''
Descripttion: 这是一个测试的main文件，所有参数默认如下：
    --batch-size 128 
    --epochs 100 
    --lr 0.001 
    --optim Adam 
    --n-conv 4 
    --atom-fea-len 64 
    --h-fea-len 256 
    --n-h 2 
    --lr-milestones 50 80 
    --val-ratio 0.15 
    --test-ratio 0.15 
    --weight-decay 1e-5 
    --print-freq 20
version: 0.0
Author: 王逸轩
Date: 2025-03-05 11:47:51
LastEditors: 王逸轩
LastEditTime: 2025-03-05 12:05:13
'''

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
from CSAT.data import CIFData
from CSAT.data import collate_pool, get_train_val_test_loader
from CSAT.model import CrystalGraphConvNet

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# python main_demo_0.py ./dataset

# 参数解析
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')

# 数据集路径
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
# 是否禁用CUDA
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
# 数据加载器工作线程数
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
# 训练轮数
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
# 起始轮数
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 批次大小
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# 学习率 0.01
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
# 学习率衰减点
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
# 动量
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# 权重衰减
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 0)')
# 打印频率
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
# 恢复训练
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

######################################################################################
train_group = parser.add_mutually_exclusive_group()
# 训练数据比例
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
# 训练数据大小
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')

########################################################################################
# 验证数据比例
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.15, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.15)')
# 验证数据大小
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')

########################################################################################
# 测试数据比例
test_group = parser.add_mutually_exclusive_group()
# 测试数据大小
test_group.add_argument('--test-ratio', default=0.15, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.15)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')
########################################################################################
# 优化器
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
# 原子特征长度
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
# 隐藏特征长度
parser.add_argument('--h-fea-len', default=256, type=int, metavar='N',
                    help='number of hidden features after pooling')
# 卷积层数
parser.add_argument('--n-conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
# 隐藏层数
parser.add_argument('--n-h', default=2, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

# 是否启用CUDA
args.cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print("CUDA", torch.cuda.is_available())

# 最佳MAE误差
best_mae_error = 1e10


# 主函数
def main():
    # 超参数、最有MAE误差初始化
    global args, best_mae_error

    # 加载数据集，输出：元组 ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    dataset = CIFData(*args.data_options, k_hop=3)
    # 将数据列表整理成用于预测晶体的批处理性能，打包(atom_fea, nbr_fea, nbr_fea_idx, target)
    collate_fn = collate_pool
    # 数据集划分函数 
    # 输入完整数据集 + 划分参数 
    # 输出：返回 train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    # 任务类型为回归
    if len(dataset) < 500:
        warnings.warn('数据集较小，预期较低的准确性。 ')
        # 从数据集中随机选择500个样本
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        # 从数据集中随机选择500个样本
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        # 从随机取样的500样本数据列表中提取数据、目标、晶体ID
        _, sample_target, _ = collate_pool(sample_data_list)
        # 用于目标值的正则化
        normalizer = Normalizer(sample_target)

    # 构建模型
    # 获取第一个样本的全部特征
    full_sample = dataset[0]
    structures = (full_sample[0][0], full_sample[0][1], full_sample[0][2])  # atom_fea, nbr_fea, nbr_fea_idx
    # 原始atom_fea
    orig_atom_fea_len = structures[0].shape[-1]
    # 邻居特征长度nbr_fea
    nbr_fea_len = structures[1].shape[-1]
    
    # 
    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=128,  # 增大特征维度
        n_conv=4,          # 增加层数
        h_fea_len=256,
        n_h=2,
        classification=False
    )
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
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
    
    for i, (input_batch, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # 正确解包输入元组
        (atom_fea, nbr_fea, nbr_fea_idx, 
         sub_nodes, sub_edges, sub_indicator) = input_batch
        
        batch_data = (
            atom_fea.to(device),
            nbr_fea.to(device),
            nbr_fea_idx.to(device),
            sub_nodes.to(device),
            sub_edges.to(device),
            sub_indicator.to(device)
        )

        # normalize target
        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(batch_data)

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

        if i % args.print_freq == 0:
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
        # === 修正输入解包方式 ===
        (atom_fea, nbr_fea, nbr_fea_idx, 
         sub_nodes, sub_edges, sub_indicator) = input
        
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        sub_nodes = sub_nodes.to(device)
        sub_edges = sub_edges.to(device)
        sub_indicator = sub_indicator.to(device)
        
         # 组合输入元组
        model_input = (
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            sub_nodes,
            sub_edges,
            sub_indicator
        )
       
        if args.cuda:
            target = target.cuda()
        target_normed = normalizer.norm(target)  # 确保归一化在相同设备
        target_var = Variable(target_normed)
        
        # compute output
        output = model(model_input)
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

        if i % args.print_freq == 0:
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
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
