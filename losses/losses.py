from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        features = input.view(input.shape[0], input.shape[1], -1)
        gram_matrix = torch.bmm(features, features.transpose(1,2))
        return gram_matrix

class DiversityLoss(nn.Module):
    def __init__(self, size_average=True, use_gram=True, reduce=True):
        super(DiversityLoss, self).__init__()
        self.use_gram = use_gram
        if use_gram:
            self.gm = GramMatrix()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input):
        if self.use_gram:
            mats = [self.gm(x) for x in input] 
        else:
            mats = [x.view(x.shape[0], -1) for x in input]
        res = 0
        count = 0
        for a in range(len(input)):
            for b in range(len(input)):
                if a != b:
                    norm_a = torch.norm(mats[a].view(mats[a].shape[0], -1), dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(mats[a])
                    norm_b = torch.norm(mats[b].view(mats[b].shape[0], -1), dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(mats[b])
                    res = res + (mats[a] * mats[b]) / (norm_a * norm_b)
                    count += 1
        res = res / count
        if not self.reduce:
            return res
        if self.size_average:
            return torch.mean(res)
        else:
            return torch.sum(res)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, normalize=True):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
        self.normalize=True

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        if self.normalize:
            euclidean_distance = 1 / float(np.prod(output1.shape[1])) * euclidean_distance
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class NormalizedMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(NormalizedMSELoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        norm_input = torch.norm(input.view(input.shape[0], -1), p=2, dim=1)
        norm_target = torch.norm(target.view(target.shape[0], -1), p=2, dim=1)
        while len(norm_input.shape) < len(input.shape):
            norm_input.unsqueeze_(-1)
            norm_target.unsqueeze_(-1)
        norm_input = norm_input.expand_as(input)
        norm_target = norm_target.expand_as(target)
        loss = F.mse_loss(input / norm_input, target / norm_target, reduce=False)
        if not self.reduce:
            return loss 
        else:
            if self.size_average:
                return torch.mean(torch.sum(loss.view(loss.shape[0], -1)))
            else:
                return torch.sum(norm_loss)
