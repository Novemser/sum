# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Train a network on multiple GPUs using multiprocessing.
"""

from itertools import cycle, islice
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from fairseq import nccl, utils
from fairseq.multiprocessing_event_loop import MultiprocessingEventLoop, Future
from fairseq.nag import NAG
from fairseq.sequence_generator import SequenceGenerator

from collections import namedtuple
 
import numpy as np
# res tuples
Results = namedtuple('Results', 
    ['loss', 'grad_norm', 'ml_loss', 
    'rl_loss', 'mean_rouge_greedy', 
    'mean_rouge_sampled', 'mean_sum_log_prob']
    )        

class MultiprocessingTrainer(MultiprocessingEventLoop):
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.

    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each process, and asynchronous functions
    (prefixed with `_async_`), which run on each process in parallel.
    """

    OPTIMIZERS = ['adagrad', 'adam', 'nag', 'sgd']

    def __init__(self, args, model, criterion, device_ids=None,
                 multiprocessing_method='spawn',
                 src_dict=None, dst_dict=None):
        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
        super().__init__(device_ids, multiprocessing_method)

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        model = model.share_memory()
        nccl_uid = nccl.get_unique_id()
        self.criterion = criterion

        Future.gen_list([
            self.call_async(rank, '_async_init', args=args, model=model,
                            criterion=criterion, nccl_uid=nccl_uid,
                            src_dict=src_dict, dst_dict=dst_dict
                            )
            for rank in range(self.num_replicas)
        ])

        self._grads_initialized = False

    def _async_init(self, rank, device_id, args, model, criterion, nccl_uid, src_dict=None, dst_dict=None):
        """Initialize child processes."""
        self.args = args

        # copy src and dst dictionary
        self.src_dict = src_dict
        self.dst_dict = dst_dict

        # copy enable rl
        self.enable_rl = args.enable_rl

        # set CUDA device
        torch.cuda.set_device(device_id)

        # initialize NCCL
        nccl.initialize(self.num_replicas, nccl_uid, device_id)

        # copy model and criterion to current device
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # initialize optimizer
        self.optimizer = self._build_optimizer()
        self.flat_grads = None
        self.loss = None

        if self.enable_rl:
            self.rl_loss = None

        # initialize LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # initialize generator
        models = [model] # SequenceGenerator accepts a list of models
        self.generator = SequenceGenerator(models, beam_size=1, minlen=args.minlen, 
            maxlen=args.max_len_b, stop_early=(not args.no_early_stop), 
            normalize_scores=(not args.unnormalized), len_penalty=args.lenpen).cuda()
        
    def _build_optimizer(self):
        if self.args.optimizer == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                    betas=eval(self.args.adam_betas),
                                    weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'nag':
            return NAG(self.model.parameters(), lr=self.args.lr,
                       momentum=self.args.momentum,
                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))

    def _build_lr_scheduler(self):
        if self.args.force_anneal > 0:
            def anneal(e):
                if e < self.args.force_anneal:
                    return 1
                else:
                    return self.args.lrshrink ** (e + 1 - self.args.force_anneal)
            lr_scheduler = LambdaLR(self.optimizer, anneal)
            lr_scheduler.best = None
        else:
            # decay the LR by 0.1 every time the validation loss plateaus
            lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=0)
        return lr_scheduler

    def get_model(self):
        """Get one of the model replicas."""
        # just return the first model, since all replicas are the same
        return self.call_async(0, '_async_get_model').gen()

    def _async_get_model(self, rank, device_id):
        return self.model

    def save_checkpoint(self, filename, extra_state):
        """Save a checkpoint for the current model."""
        self.call_async(0, '_async_save_checkpoint', filename=filename, extra_state=extra_state).gen()

    def _async_save_checkpoint(self, rank, device_id, filename, extra_state):
        utils.save_state(filename, self.args, self.model, self.criterion, self.optimizer,
                         self.lr_scheduler, self._optim_history, extra_state)

    def load_checkpoint(self, filename):
        """Load a checkpoint into the model replicas in each process."""
        results = Future.gen_list([
            self.call_async(rank, '_async_load_checkpoint', filename=filename)
            for rank in range(self.num_replicas)
        ])
        extra_state = results[0]
        return extra_state

    def _async_load_checkpoint(self, rank, device_id, filename):
        extra_state, self._optim_history = utils.load_state(
            filename, self.model, self.criterion, self.optimizer,
            self.lr_scheduler, args=self.args, cuda_device=device_id)
        return extra_state

    def set_seed(self, seed):
        Future.gen_list([
            self.call_async(rank, '_async_set_seed', seed=seed)
            for rank in range(self.num_replicas)
        ])

    def _async_set_seed(self, rank, device_id, seed):
        torch.manual_seed(seed)

    def generate(self, input):
        """
        Generate greedy and sampled outputs
        """
        def lstrip_pad(tensor):
            return tensor[tensor.eq(self.generator.pad).sum():]
        
        args = self.args
        srclen = input['src_tokens'].size(1)
        sampled_hypos = self.generator.generate(input['src_tokens'], input['src_positions'],
                             maxlen=(args.max_len_a*srclen + args.max_len_b), 
                             enable_sample=True)
        greedy_hypos = self.generator.generate(input['src_tokens'], input['src_positions'],
                              maxlen=(args.max_len_a*srclen + args.max_len_b), 
                              enable_sample=False)
        
        
        #greedy_hypos = sampled_hypos
        ref_hypo_res = [] # [(ref_str, greedy_hypo_str, sampled_hypo_str)]
        for i, id in enumerate(self._sample['id']):
            src = input['src_tokens'].data[i, :]
            # remove padding from ref, which appears at the beginning
            ref = lstrip_pad(self._sample['target'].data[i, :])
            greedy_hypo = greedy_hypos[i]
            sampled_hypo = sampled_hypos[i]
            
            ref = ref.int().cpu()

            # we don't need sum_log_probs for greedy output
            ref_str, greedy_hypo_str, _ = utils.display_hypotheses(id, ref, greedy_hypo[:min(len(greedy_hypo), args.nbest)], 
                self.src_dict, self.dst_dict, args=self.args)
            _, sampled_hypo_str, _sum_log_probs = utils.display_hypotheses(id, ref, sampled_hypo[:min(len(sampled_hypo), args.nbest)], 
                self.src_dict, self.dst_dict, args=self.args)
            #print('----------')
            #print('ref: {}\n greedy_hypo: {}\n sampled_hypo: {}'.format(ref_str, greedy_hypo_str, sampled_hypo_str))
            ref_hypo_res.append((ref_str, greedy_hypo_str[0], 
                                 sampled_hypo_str[0], 
                                 _sum_log_probs[0],
                                 sampled_hypo[0]['tokens'])) # beam_size = 1
        
        return ref_hypo_res

    def train_step(self, samples):
        """Do forward, backward and gradient step in parallel."""
        # PyTorch initializes gradient buffers lazily, so the first
        # train step needs to send non-empty samples to all replicas
        replace_empty_samples = False
        if not self._grads_initialized:
            replace_empty_samples = True
            self._grads_initialized = True

        # scatter sample across GPUs
        self._scatter_samples(samples, replace_empty_samples=replace_empty_samples)

        # forward pass
        sample_sizes, logging_outputs = Future.gen_tuple_list([
            self.call_async(rank, '_async_forward')
            for rank in range(self.num_replicas)
        ])

        # backward pass, all-reduce gradients and take an optimization step
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        grad_norms = Future.gen_list([
            self.call_async(rank, '_async_backward_and_opt', grad_denom=grad_denom)
            for rank in range(self.num_replicas)
        ])

        # aggregate logging output
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)
        logging_output['gnorm'] = grad_norms[0]  # log the gradient norm

        ## TODO: add more logging info such as rl loss,etc.
        return logging_output

    def _async_forward(self, rank, device_id, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        if self._sample is None:
            return 0, {}

        # calculate loss and sample size
        self.loss, sample_size, logging_output = self.criterion(self.model, self._sample)

        if not eval and self.enable_rl:
            args = self.args
            # update generator
            models = [self.model] # SequenceGenerator accepts a list of models
            self.generator.models = models ## TODO: may be not necessary to copy...
            input = self._sample['net_input']
            
            ref_hypo_res = self.generate(input)
            refs = [item[0] for item in ref_hypo_res]
            greedy_sums = [item[1] for item in ref_hypo_res]
            sampled_sums = [item[2] for item in ref_hypo_res]
            sum_log_probs = [item[3] for item in ref_hypo_res]
            sampled_tokens = [item[4]for item in ref_hypo_res]
            fmt = 'ref: {}\n'
            fmt += 'greedy: {}\n'
            fmt += 'sampled: {}'
            print(fmt.format(refs[0], greedy_sums[0], sampled_sums[0]))
            seq_lens = torch.Tensor([seq.size()[0] for seq in sampled_tokens]).cuda()
            sum_log_probs = torch.cat(sum_log_probs)
            
            # evaluate rouge
            rouge_greedy = torch.Tensor([utils.evaluate([greedy_sums[i]], [refs[i]]) for i in range(len(refs))])
            rouge_sampled = torch.Tensor([utils.evaluate([sampled_sums[i]], [refs[i]]) for i in range(len(refs))])
            rouge_delta = rouge_greedy - rouge_sampled
            #rouge_delta =  - rouge_sampled
            
            # compute rl loss
            rl_loss = Variable(rouge_delta.cuda(), requires_grad=False) * sum_log_probs
            self.rl_loss = torch.sum(rl_loss) / torch.sum(Variable(seq_lens, requires_grad=False))

            # compute hybrid loss
            ml_loss = self.loss
            self.loss = args.loss_scale * self.rl_loss + (1 - args.loss_scale) * ml_loss

            # compute mean statistics
            mean_rouge_greedy = sum(rouge_greedy)/len(rouge_greedy)
            mean_rouge_sampled = sum(rouge_sampled)/len(rouge_sampled)
            mean_sum_log_prob = torch.sum(sum_log_probs)/torch.sum(Variable(seq_lens, requires_grad=False))
            mean_sum_log_prob = mean_sum_log_prob.data[0]
            print(mean_rouge_greedy, mean_rouge_sampled, mean_sum_log_prob)
        return sample_size, logging_output

    def _async_backward_and_opt(self, rank, device_id, grad_denom):
        if self.loss is not None:
            # backward pass
            self.loss.backward()

        # flatten grads into a contiguous block of memory
        if self.flat_grads is None:
            self.flat_grads = self._flatten_grads_(self.model)

        # all-reduce grads
        nccl.all_reduce(self.flat_grads)

        # normalize grads
        if grad_denom != 0:
            self.flat_grads.div_(grad_denom)

        # clip grads
        grad_norm = self._clip_grads_(self.flat_grads, self.args.clip_norm)

        # take an optimization step
        self.optimizer.step()

        # reset loss
        self.loss = None
        if self.enable_rl:
            self.rl_loss = None

        return grad_norm

    def _flatten_grads_(self, model):
        num_params = sum(p.data.numel() for p in model.parameters())
        flat_grads = next(model.parameters()).data.new(num_params)
        offset = 0
        for p in model.parameters():
            grad = p.grad.data
            numel, sz = grad.numel(), grad.size()
            flat_grads[offset:offset+numel] = grad.view(-1)
            grad.set_(flat_grads[offset:offset+numel])
            grad.resize_(sz)  # preserve original shape
            offset += numel
        return flat_grads

    def _clip_grads_(self, flat_grads, clipv):
        norm = flat_grads.norm()
        if clipv > 0 and norm > clipv:
            coef = max(norm, 1e-6) / clipv
            flat_grads.div_(coef)
        return norm

    def valid_step(self, samples):
        """Do forward pass in parallel."""
        # scatter sample across GPUs
        self._scatter_samples(samples, volatile=True)

        # forward pass
        _sample_sizes, logging_outputs = Future.gen_tuple_list([
            self.call_async(rank, '_async_forward', eval=True)
            for rank in range(self.num_replicas)
        ])

        # aggregate logging output
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)

        return logging_output

    def get_lr(self):
        """Get the current learning rate."""
        return self.call_async(0, '_async_get_lr').gen()

    def _async_get_lr(self, rank, device_id):
        return self.optimizer.param_groups[0]['lr']

    def lr_step(self, val_loss=None, epoch=None):
        """Adjust the learning rate depending on the validation loss."""
        lr = Future.gen_list([
            self.call_async(rank, '_async_lr_step', val_loss=val_loss, epoch=epoch)
            for rank in range(self.num_replicas)
        ])
        return lr[0]

    def _async_lr_step(self, rank, device_id, epoch, val_loss):
        # update the learning rate
        if self.args.force_anneal > 0:
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step(val_loss, epoch)
        return self.optimizer.param_groups[0]['lr']

    def _scatter_samples(self, samples, volatile=False, replace_empty_samples=False):
        """Split and distribute a sample across GPUs."""
        if not replace_empty_samples:
            # pad with None until its size is equal to the number of replicas
            samples = samples + [None]*(self.num_replicas - len(samples))
        else:
            # pad by cycling through the given samples
            samples = list(islice(cycle(samples), self.num_replicas))

        Future.gen_list([
            self.call_async(rank, '_async_prepare_sample', sample=samples[rank], volatile=volatile)
            for rank in range(self.num_replicas)
        ])

    def _async_prepare_sample(self, rank, device_id, sample, volatile):
        if sample is None:
            self._sample = None
        else:
            self._sample = utils.prepare_sample(sample, volatile=volatile, cuda_device=device_id)
