# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import logging
import os
import torch
import traceback

from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import criterions, models

from fairseq.rouge import rouge


def parse_args_and_arch(parser):
    args = parser.parse_args()
    args.model = models.arch_model_map[args.arch]
    args = getattr(models, args.model).parse_arch(args)
    return args


def build_model(args, src_dict, dst_dict):
    assert hasattr(models, args.model), 'Missing model type'
    return getattr(models, args.model).build_model(args, src_dict, dst_dict)


def build_criterion(args, src_dict, dst_dict):
    padding_idx = dst_dict.pad()
    if args.label_smoothing > 0:
        return criterions.LabelSmoothedCrossEntropyCriterion(args.label_smoothing, padding_idx)
    else:
        return criterions.CrossEntropyCriterion(padding_idx)


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, args, model, criterion, optimizer, lr_scheduler, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    prefix_to_remove = 'decoder._orig' # TODO: better handle such case
    copy_dict = model.state_dict()
    keys = list(copy_dict.keys())
    
    for k in keys:
        # print(k)
        if prefix_to_remove in k:
            # print('removed: {}'.format(k))
            del copy_dict[k]
            # print(k in copy_dict)

    state_dict = {
        'args': args,
        'model': model.state_dict(),
        # 'model': copy_dict,
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer': optimizer.state_dict(),
                'best_loss': lr_scheduler.best,
            }
        ],
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_state(filename, model, criterion, optimizer, lr_scheduler, args=None, cuda_device=None):
    if not os.path.exists(filename):
        return None, []
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )
    state = _upgrade_state_dict(state)

    # load model parameters
    model.load_state_dict(state['model'])

    # only load optimizer and lr_scheduler if they match with the checkpoint
    optim_history = state['optimizer_history']
    last_optim = optim_history[-1]
    if last_optim['criterion_name'] == criterion.__class__.__name__:
        optimizer.load_state_dict(last_optim['optimizer'])
        lr_scheduler.best = last_optim['best_loss']
        # hard set learning rate when needed
        if args and args.hardset_lr:
            optimizer.param_groups[0]['lr'] = args.lr

    return state['extra_state'], optim_history


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': criterions.CrossEntropyCriterion.__name__,
                'optimizer': state['optimizer'],
                'best_loss': state['best_loss'],
            },
        ]
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    return state


def load_ensemble_for_inference(filenames, src_dict, dst_dict):
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        states.append(
            torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        )
    args = states[0]['args']

    # build ensemble
    ensemble = []
    for state in states:
        model = build_model(args, src_dict, dst_dict)
        model.load_state_dict(state['model'])
        ensemble.append(model)
    return ensemble


def prepare_sample(sample, volatile=False, cuda_device=None):
    """Wrap input tensors in Variable class."""

    def make_variable(tensor):
        if cuda_device is not None and torch.cuda.is_available():
            tensor = tensor.cuda(async=True, device=cuda_device)
        return Variable(tensor, volatile=volatile)

    return {
        'id': sample['id'],
        'ntokens': sample['ntokens'],
        'target': make_variable(sample['target']),
        'net_input': {
            key: make_variable(sample[key])
            for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions']
        },
    }

def evaluate(hypotheses, references, metric='rouge_l/f_score'):
    """
    summary: []
    reference: []
    """
    scores = rouge(hypotheses, references)
    return scores[metric].item()

def to_token(dict, i, runk):
    return runk if i == dict.unk() else dict[i]

def unk_symbol(dict, ref_unk=False):
    return '<{}>'.format(dict.unk_word) if ref_unk else dict.unk_word

def to_sentence(dict, tokens, bpe_symbol=None, ref_unk=False):
    if torch.is_tensor(tokens) and tokens.dim() == 2:
        sentences = [to_sentence(dict, token) for token in tokens]
        return '\n'.join(sentences)
    eos = dict.eos()
    runk = unk_symbol(dict, ref_unk=ref_unk)
    sent = ' '.join([to_token(dict, i, runk) for i in tokens if i != eos])
    if bpe_symbol is not None:
        sent = sent.replace(bpe_symbol, '')
    return sent

'''
def _display_hypotheses(id, src, orig, ref, hypos, src_dict, dst_dict):
    """
    Dispaly hypos with bpe symbol always removed
    """
    bpe_symbol = '@@'
    id_str = '' if id is None else '-{}'.format(id)
    hypo_str = []
    sum_log_probs = []
    # print('S{}\t{}'.format(id_str, src_str))
    if orig is not None:
        print('O{}\t{}'.format(id_str, orig.strip()))
    ref_str = to_sentence(dst_dict, ref, bpe_symbol, ref_unk=True)
    for hypo in hypos:
        hypo_str.append(to_sentence(dst_dict, hypo['tokens'], bpe_symbol))
        sum_log_probs.append(hypo['sum_log_prob'])
        # if args.unk_replace_dict != '':
        #    hypo_str = replace_unk(hypo_str, align_str, orig, unk_symbol(dst_dict))
    return ref_str, hypo_str, sum_log_probs
'''

def display_hypotheses(id, ref, hypos, src_dict, dst_dict, args=None):
        ref_str = dst_dict.string(ref, args.remove_bpe, escape_unk=True)
        hypo_str = []
        sum_log_probs = []
        for hypo in hypos:
            hypo_str.append(dst_dict.string(hypo['tokens'], args.remove_bpe))
            sum_log_probs.append(hypo['sum_log_prob'])
        return ref_str, hypo_str, sum_log_probs
            
