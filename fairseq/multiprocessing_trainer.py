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

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from fairseq import nccl, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.criterions import FairseqCriterion
from fairseq.multiprocessing_event_loop import MultiprocessingEventLoop, Future
from fairseq.nag import NAG

import copy

from fairseq.rouge import rouge

class MultiprocessingTrainer(MultiprocessingEventLoop):
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.

    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each process, and asynchronous functions
    (prefixed with `_async_`), which run on each process in parallel.
    """

    def __init__(self, args, model, device_ids=None,
                 multiprocessing_method='spawn', src_dict=None, dst_dict=None):
        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
        super().__init__(device_ids, multiprocessing_method)

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        model = model.share_memory()
        nccl_uid = nccl.get_unique_id()

        Future.gen_list([
            self.call_async(rank, '_async_init', args=args, model=model, 
                            src_dict=src_dict, dst_dict=dst_dict,
                            nccl_uid=nccl_uid)
            for rank in range(self.num_replicas)
        ])
            
        self.enable_rl = args.enable_rl
        self.args = args

    def _async_init(self, rank, device_id, args, model, nccl_uid, src_dict=None, dst_dict=None):
        """Initialize child processes."""
        self.args = args

        # set torch.seed in this process
        torch.manual_seed(args.seed)

        # set CUDA device
        torch.cuda.set_device(device_id)

        # initialize NCCL
        nccl.initialize(self.num_replicas, nccl_uid, device_id)

        # copy model to current device
        self.model = model.cuda()

        # initialize optimizer
        self.optimizer = NAG(self.model.parameters(), lr=self.args.lr,
                             momentum=self.args.momentum,
                             weight_decay=self.args.weight_decay)
        self.flat_grads = None

        # initialize LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.enable_rl = args.enable_rl
        self.generator = None
        self.args = args
        if self.enable_rl:
            # Initialize generator
            models = [model] # SequenceGenerator accepts a list of models
            self.generator = SequenceGenerator(models, dst_dict, beam_size=1,
                                           stop_early=(not args.no_early_stop),
                                           normalize_scores=(not args.unnormalized),
                                           len_penalty=args.lenpen).cuda()

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

    def save_checkpoint(self, args, epoch, batch_offset, val_loss=None):
        """Save a checkpoint for the current model."""
        self.call_async(0, '_async_save_checkpoint', args=args, epoch=epoch,
                        batch_offset=batch_offset, val_loss=val_loss).gen()

    def _async_save_checkpoint(self, rank, device_id, args, epoch, batch_offset, val_loss):
        utils.save_checkpoint(args, epoch, batch_offset, self.model,
                              self.optimizer, self.lr_scheduler, val_loss)

    def load_checkpoint(self, filename):
        """Load a checkpoint into the model replicas in each process."""
        results = Future.gen_list([
            self.call_async(rank, '_async_load_checkpoint', filename=filename)
            for rank in range(self.num_replicas)
        ])
        epoch, batch_offset = results[0]
        return epoch, batch_offset

    def _async_load_checkpoint(self, rank, device_id, filename):
        return utils.load_checkpoint(filename, self.model, self.optimizer,
                                     self.lr_scheduler, cuda_device=device_id)

    def train_step(self, samples, criterion):
        """Do forward, backward and gradient step in parallel."""
        assert isinstance(criterion, FairseqCriterion)
        
        # scatter sample across GPUs
        self._scatter_samples(samples)
        criterion.prepare(samples)

        # forward pass, backward pass and gradient step
        losses = [
            self.call_async(rank, '_async_train_step', criterion=criterion)
            for rank in range(self.num_replicas)
        ]

        # aggregate losses and gradient norms
        losses, grad_norms = Future.gen_tuple_list(losses)
        loss = criterion.aggregate(losses)

        return loss, grad_norms[0]

    def _async_train_step(self, rank, device_id, criterion):
        self.model.train()

        # If enable_rl, generate two outputs in inference model
        # 1) greedy
        # 2) sampled
        if self.enable_rl:
            args = self.args
            # since deepcopy does not support our case, we do not use fast generation
            # cur_model = copy.deepcopy(self.model) # deep copy current model, since once made generation fast, cannot be trained
            # cur_model.make_generation_fast_(1, not args.no_beamable_mm) # for fast generation
            self.generator.models = [self.model]       
            input = self._sample['net_input']
            srclen = input['src_tokens'].size(1)
            
            def generate():
                """
                Generate greedy and sampled outputs
                """
                def lstrip_pad(tensor):
                    pad = self.generator.pad
                    return tensor[tensor.eq(pad).sum():]
                
                greedy_hypos = self.generator.generate(input['src_tokens'], input['src_positions'],
                                      maxlen=(args.max_len_a*srclen + args.max_len_b), 
                                      enable_sample=False)
                sampled_hypos = self.generator.generate(input['src_tokens'], input['src_positions'],
                                     maxlen=(args.max_len_a*srclen + args.max_len_b), 
                                     enable_sample=True)
                
                ref_hypo_res = [] # [(ref_str, greedy_hypo_str, sampled_hypo_str)]
                for i, id in enumerate(self._sample['id']):
                    src = input['src_tokens'].data[i, :]
                    # remove padding from ref, which appears at the beginning
                    ref = lstrip_pad(self._sample['target'].data[i, :])
                    greedy_hypo = greedy_hypos[i]
                    sampled_hypo = sampled_hypos[i]

                    ref = ref.int().cpu()
                    # we don't need sum_log_probs for greedy output
                    ref_str, greedy_hypo_str, _ = utils.display_hypotheses(id, src, None, ref, 
                                                                         greedy_hypo[:min(len(greedy_hypo), args.nbest)],
                                                                         self.src_dict, self.dst_dict)
                    _, sampled_hypo_str, _sum_log_probs = utils.display_hypotheses(id, src, None, ref, 
                                                                         sampled_hypo[:min(len(sampled_hypo), args.nbest)],
                                                                         self.src_dict, self.dst_dict)
                    ref_hypo_res.append((ref_str, greedy_hypo_str[0], sampled_hypo_str[0], _sum_log_probs[0])) # beam_size = 1

                return ref_hypo_res
            ref_hypo_res = generate()
            refs = [item[0] for item in ref_hypo_res]
            greedy_sums = [item[1] for item in ref_hypo_res]
            sampled_sums = [item[2] for item in ref_hypo_res]
            sum_log_probs = [item[3] for item in ref_hypo_res]

            def evaluate(hypotheses, references, metric='rouge_l/f_score'):
                """
                summary: []
                reference: []
                """
                scores = rouge(hypotheses, references)
                return scores[metric].item()

            rouge_greedy = [evaluate([greedy_sums[i]], [refs[i]]) for i in range(len(refs))]
            rouge_sampled = [evaluate([sampled_sums[i]], [refs[i]]) for i in range(len(refs))]
            
            
            rl_loss = 0
            
            for r_g, r_s, sum_log_prob in zip(rouge_greedy, rouge_sampled, sum_log_probs):
                rl_loss += (r_g - r_s) * sum_log_prob
            rl_loss /= len(rouge_greedy) # normalized by # sentences

        # zero grads even if net_input is None, since we will all-reduce them
        self.optimizer.zero_grad()

        # calculate loss and grads
        loss = 0
        if self._sample is not None:
            net_output = self.model(**self._sample['net_input'])
            ml_loss = criterion(net_output, self._sample)
            if self.enable_rl:
                loss_ = args.loss_scale * rl_loss + (1 - args.loss_scale) * ml_loss
                print('\n mixed_loss: {:^10.4f}, ml_loss: {:^10.4f}, rl_loss: {:^10.4f}, mean_rouge_greedy: {:^10.4f}, mean_rouge_sampled: {:^10.4f}'.format(
                        loss_.data[0],
                        ml_loss.data[0],
                        rl_loss,
                        sum(rouge_greedy)/len(rouge_greedy), 
                        sum(rouge_sampled)/len(rouge_sampled)))
            else:
                loss_ = ml_loss
            loss_.backward()
            loss = loss_.data[0]

        # flatten grads into a contiguous block of memory
        if self.flat_grads is None:
            self.flat_grads = self._flatten_grads_(self.model)

        # all-reduce grads
        nccl.all_reduce(self.flat_grads)

        # clip grads
        grad_norm = self._clip_grads_(self.flat_grads, self.args.clip_norm)

        # take an optimization step
        self.optimizer.step()

        return loss, grad_norm

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

    def valid_step(self, samples, criterion):
        """Do forward pass in parallel."""
        # scatter sample across GPUs
        self._scatter_samples(samples, volatile=True)
        criterion.prepare(samples)

        # forward pass
        losses = [
            self.call_async(rank, '_async_valid_step', criterion=criterion)
            for rank in range(self.num_replicas)
        ]

        # aggregate losses
        loss = criterion.aggregate(Future.gen_list(losses))

        return loss

    def _async_valid_step(self, rank, device_id, criterion):
        if self._sample is None:
            return 0
        self.model.eval()
        net_output = self.model(**self._sample['net_input'])
        loss = criterion(net_output, self._sample)
        return loss.data[0]

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

    def _scatter_samples(self, samples, volatile=False):
        """Split and distribute a sample across GPUs."""
        # Pad with None until its size is equal to the number of replicas.
        samples = samples + [None]*(self.num_replicas - len(samples))

        Future.gen_list([
            self.call_async(rank, '_async_prepare_sample', sample=samples[rank], volatile=volatile)
            for rank in range(self.num_replicas)
        ])

    def _async_prepare_sample(self, rank, device_id, sample, volatile):
        if sample is None:
            self._sample = None
        else:
            '''
            self._sample: {'id': LongTensor[181], 
                            'ntokens': 1314, 
                            'target': LongTensor[181*12], 
                            'net_input': {
                                'src_token': LongTensor[181*12],
                                'src_position': LongTensor[181*12],
                                'input_tokens': LongTensor[181*12],
                                'input_position': LongTensor[181*12]
                                }
                            }
            '''
            self._sample = utils.prepare_sample(sample, volatile=volatile, cuda_device=device_id)
