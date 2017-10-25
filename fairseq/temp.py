import torch
from torch.autograd import Variable


def sample(self, src_tokens, src_positions, beam_size=None, maxlen=None):
    bsz = src_tokens.size(0)
    beam_size = beam_size if beam_size is not None else self.beam_size
    maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

    assert beam_size == 1
    encoder_outs = []
    for model in self.models:
        model.eval()
        model.decoder.clear_incremental_state()  # start a fresh sequence

        # compute the encoder output and expand to beam size
        encoder_out = model.encoder(src_tokens, src_positions)
        encoder_out = self._expand_encoder_out(encoder_out, beam_size)
        encoder_outs.append(encoder_out)

    # initialize buffers
    tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
    tokens[:, 0] = self.eos
    seq_log_probs = Variable(encoder_outs[0][0].data.new(bsz * beam_size, maxlen + 1).fill_(0))

    # list of finalized sentences
    finalized = [[] for i in range(bsz)]
    # number of candidate hypos per step
    cand_size = 1

    # helper function for allocating buffers on the fly
    buffers = {}
    def buffer(name, type_of=tokens):
        if name not in buffers:
            buffers[name] = type_of.new()
        return buffers[name]

    for step in range(maxlen + 1):  # one extra step for EOS marker
        probs, avg_attn_scores = self._decode(tokens[:, :step+1], encoder_outs)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()

        cand_indices = buffer('cand_indices')

        torch.multinomial(probs.view(bsz, -1).data.exp(), cand_size, replacement=False, out=cand_indices)
        sampled_log_prob = probs.gather(1, Variable(cand_indices, requires_grad=False))
        seq_log_probs[:, step] = sampled_log_prob.view(-1)
        
        tokens[:, step+1] = cand_indices.view(-1)
        
    for i in range(bsz):
        finalized[i].append(
                {
                    'token': tokens[i, :],
                    'sampled_log_prob': seq_log_probs[i, :]
                })
        
    return finalized