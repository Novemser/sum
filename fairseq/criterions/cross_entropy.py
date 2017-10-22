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


class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        input = net_output.view(-1, net_output.size(-1))
        target = sample['target'].view(-1)
        loss = F.cross_entropy(input, target, size_average=False, ignore_index=self.padding_idx)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data[0],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

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
