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

from fairseq import criterions, data, models

from fairseq.rouge import rouge


def parse_args_and_arch(parser):
    args = parser.parse_args()
    args.model = models.arch_model_map[args.arch]
    args = getattr(models, args.model).parse_arch(args)
    return args


def build_model(args, dataset):
    assert hasattr(models, args.model), 'Missing model type'
    return getattr(models, args.model).build_model(args, dataset)


def build_criterion(args, dataset):
    padding_idx = dataset.dst_dict.pad()
    if args.label_smoothing > 0:
        return criterions.LabelSmoothedCrossEntropyCriterion(args.label_smoothing, padding_idx)
    else:
        return criterions.CrossEntropyCriterion(padding_idx)


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except:
            if i == 2:
                logging.error(traceback.format_exc())


def save_checkpoint(args, epoch, batch_offset, model, optimizer, lr_scheduler, val_loss=None):
    prefix_to_remove = 'decoder._orig' # TODO: better handle such case
    copy_dict = model.state_dict()
    keys = list(copy_dict.keys())
    
    for k in keys:
        # print(k)
        if prefix_to_remove in k:
            # print('removed: {}'.format(k))
            del copy_dict[k]
            # print(k in copy_dict)
    
    # print(copy_dict.keys())       
    state_dict = {
        'args': args,
        'epoch': epoch,
        'batch_offset': batch_offset,
        # 'model': model.state_dict(),
        'model': copy_dict,
        'optimizer': optimizer.state_dict(),
        'best_loss': lr_scheduler.best,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, 'checkpoint{}.pt'.format(epoch))
            torch_persistent_save(state_dict, epoch_filename)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
            torch_persistent_save(state_dict, best_filename)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    torch_persistent_save(state_dict, last_filename)


def load_checkpoint(filename, model, optimizer, lr_scheduler, args=None,  cuda_device=None):
    if not os.path.exists(filename):
        return 1, 0
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )
    
    '''
    print(set(state['model'].keys()) - set(model.state_dict().keys()))
    print('-----')
    print(model.state_dict().keys())
    '''
    
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if args and args.hardset_lr:
        optimizer.param_groups[0]['lr'] = args.lr
    lr_scheduler.best = state['best_loss']
    epoch = state['epoch'] + 1
    batch_offset = state['batch_offset']

    gpu_str = ' on GPU #{}'.format(cuda_device) if cuda_device is not None else ''
    print('| loaded checkpoint {} (epoch {}){}'.format(filename, epoch, gpu_str))
    return epoch, batch_offset


def load_ensemble_for_inference(filenames, data_path):
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        states.append(
            torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        )

    # load dataset
    args = states[0]['args']
    dataset = data.load(data_path, args.source_lang, args.target_lang)

    # build models
    ensemble = []
    for state in states:
        model = build_model(args, dataset)
        model.load_state_dict(state['model'])
        ensemble.append(model)

    return ensemble, dataset


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

def display_hypotheses(id, src, orig, ref, hypos, src_dict, dst_dict):
    """
    Dispaly hypos with bpe symbol always removed
    """
    bpe_symbol = '@@'
    id_str = '' if id is None else '-{}'.format(id)
    hypo_str = []
    seq_log_probs = []
    # print('S{}\t{}'.format(id_str, src_str))
    if orig is not None:
        print('O{}\t{}'.format(id_str, orig.strip()))
    ref_str = to_sentence(dst_dict, ref, bpe_symbol, ref_unk=True)
    for hypo in hypos:
        hypo_str.append(to_sentence(dst_dict, hypo['tokens'], bpe_symbol))
        seq_log_probs.append(hypo['log_prob'])
        # if args.unk_replace_dict != '':
        #    hypo_str = replace_unk(hypo_str, align_str, orig, unk_symbol(dst_dict))
    return ref_str, hypo_str, seq_log_probs

def evaluate(hypotheses, references, metric='rouge_l/f_score'):
    """
    summary: []
    reference: []
    """
    scores = rouge(hypotheses, references)
    return scores[metric].item()

def sum_if_not_none(x):
    """
    Sum x if it does not contain None
    """
    s = 0
    for i in x:
        if i is None:
            return None
        s += i
        return s