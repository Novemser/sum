# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch.nn.functional as F

from .fairseq_criterion import FairseqCriterion

import torch


class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, padding_idx,enable_topic):
        super().__init__()
        self.padding_idx = padding_idx
        self.enable_topic = enable_topic

    def prepare(self, samples):
        self.denom = sum(s['ntokens'] if s else 0 for s in samples)

    def forward(self, net_output, sample):
        ###print("net_output.size():"+str(net_output.size())) ###net_output.size():torch.Size([1296, 8789])
        
        target = sample['target'].view(-1)  ###CrossEntropyCriterion target:1296
        
        if self.enable_topic :
            ###net_output_ = net_output[0] + net_output[1]
            input = net_output[0].view(-1, net_output[0].size(-1)) ###no softmax yet  1296x8789
            input_topic = net_output[1].view(-1, net_output[0].size(-1))
            topic_words_mask = net_output[2]
            input_exp = torch.exp(input) + torch.exp(input_topic) * torch.autograd.Variable(topic_words_mask.expand(input_topic.size(0), topic_words_mask.size(0)), requires_grad=False)
            input_softmax = input_exp / torch.sum(input_exp,-1)
            loss = F.nll_loss(input_softmax, target, size_average=False, ignore_index=self.padding_idx)
        else:        
            input = net_output.view(-1, net_output.size(-1)) ###no softmax yet  1296x8789
            loss = F.cross_entropy(input, target, size_average=False, ignore_index=self.padding_idx)   ###self.padding_idx:1

        return loss / self.denom

    def aggregate(self, losses):
        return sum(losses) / math.log(2)
