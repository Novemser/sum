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
from .utils import aggregate

class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, padding_idx,enable_topic):
        super().__init__()
        self.padding_idx = padding_idx
        self.enable_topic = enable_topic

    def prepare(self, samples):
        self.denom = sum(s['ntokens'] if s else 0 for s in samples)

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        ###print("net_output.size():"+str(net_output.size())) ###net_output.size():torch.Size([1296, 8789])
        
        target = sample['target'].view(-1)  ###CrossEntropyCriterion target:1296
        net_output = model(**sample['net_input'])
        
        if self.enable_topic :
            ###net_output_ = net_output[0] + net_output[1]
            input = net_output[0].view(-1, net_output[0].size(-1)) ###no softmax yet  1296x8789
            input_topic = net_output[1].view(-1, net_output[0].size(-1))
            topic_words_mask = net_output[2]
            input_exp = torch.exp(input) + torch.exp(input_topic) * torch.autograd.Variable(topic_words_mask.expand(input_topic.size(0), topic_words_mask.size(0)), requires_grad=False)
            input_softmax = input_exp / torch.sum(input_exp,-1).view(input_exp.size(0),1).expand(input_exp.size(0),input_exp.size(1))
            ###print("torch.sum(input_exp,-1):"+str(torch.sum(input_exp,-1)[0:10]))
            loss = F.nll_loss(torch.log(input_softmax), target, size_average=False, ignore_index=self.padding_idx)
            ###print("loss:"+str(loss))
        else:        
            input = net_output.view(-1, net_output.size(-1)) ###no softmax yet  1296x8789
            loss = F.cross_entropy(input, target, size_average=False, ignore_index=self.padding_idx)   ###self.padding_idx:1

        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
        #return loss / self.denom

    def aggregate(self, losses):
        return sum(losses) / math.log(2)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': aggregate(logging_outputs, 'loss', default=0, avg=False) / sample_size / math.log(2),
            'mean_rouge_greedy': aggregate(logging_outputs, 'mean_rouge_greedy', default=0, avg=True),
            'mean_rouge_sampled': aggregate(logging_outputs, 'mean_rouge_sampled', default=0, avg=True),
            'mean_sum_log_prob': aggregate(logging_outputs, 'mean_sum_log_prob', default=0, avg=True),
        }
