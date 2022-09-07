# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sd(x):
    return np.std(x, axis=0, ddof=1)


def sd_gpu(x):
    return torch.std(x, dim=0)


def normalize_gpu(x):
    x = F.normalize(x, p=1, dim=1)
    return x


def normalize(x):
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0) # 256
    r = x.size(1) # 512
    x = x.view(n, r, 1) #(256,512,1)
    c = x.size(2) # 1
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c))) # (1,1) 默认num_f=1
        b = 2 * np.pi * torch.rand(size=(r, num_f))    # (512,1)
        b = b.repeat((n, 1, 1))                        # (256,512,1)

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())   # (1,1)

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    # 归一化
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


def lossc(inputs, target, weight):
    loss = nn.NLLLoss(reduce=False)
    return loss(inputs, target).view(1, -1).mm(weight).view(1)


def cov(x, w=None):
    if w is None:
        n = x.shape[0] # 256
        cov = torch.matmul(x.t(), x) / n ()
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        # x (256,512)
        w = w.view(-1, 1) # (256,1)
        cov = torch.matmul((w * x).t(), x) # (512,512)
        e = torch.sum(w * x, dim=0).view(-1, 1) #(512,1)
        res = cov - torch.matmul(e, e.t())

    return res


def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda() # (256,512,1)
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda() # (256,1)
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i] # (256,512)
        cov1 = cov(cfeaturec, weight) # 作用为计算互协方差
        cov_matrix = cov1 * cov1 # 平方
        print(cov_matrix.shape)
        # 不算自己和自己
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss


def lossq(cfeatures, cfs):
    return - cfeatures.pow(2).sum(1).mean(0).view(1) / cfs


def lossn(cfeatures):
    return cfeatures.mean(0).pow(2).mean(0).view(1)


if __name__ == '__main__':
    pass
