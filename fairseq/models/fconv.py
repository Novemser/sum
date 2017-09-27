# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fairseq.modules import BeamableMM, LinearizedConvolution


class FConvModel(nn.Module):
    def __init__(self, encoder, decoder, padding_idx=1):
        super(FConvModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.num_attention_layers = sum([layer is not None for layer in decoder.attention])
        self.padding_idx = padding_idx
        self._is_generation_fast = False

    def forward(self, src_tokens, src_positions, input_tokens, input_positions):
        encoder_out = self.encoder(src_tokens, src_positions)
        decoder_out = self.decoder(input_tokens, input_positions, encoder_out)
        return decoder_out.view(-1, decoder_out.size(-1))

    def make_generation_fast_(self, beam_size, use_beamable_mm=False):
        """Optimize model for faster generation.

        Optimizations include:
        - remove WeightNorm
        - (optionally) use BeamableMM in attention layers

        The optimized model should not be used again for training.

        Note: this can be combined with incremental inference in the Decoder for
        even faster generation.
        """
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

        # use BeamableMM in attention layers
        if use_beamable_mm:
            self.decoder._use_beamable_mm(beam_size)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train


class Encoder(nn.Module):
    """Convolutional encoder"""
    def __init__(self, num_embeddings, embed_dim=512, max_positions=1024,
                 convolutions=((512, 3),) * 20, dropout=0.1, padding_idx=1,vocab_topic_emb=None):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)
        
        #self.embed_tokens_topic = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        #self.embed_tokens_topic.weight = nn.Parameter(vocab_topic_emb)
        
        print("Encoder:padding_idx:"+str(padding_idx))
        print("Encoder:self.embed_tokens:"+str(self.embed_tokens))
        print("Encoder:self.embed_positions:"+str(self.embed_positions))
        """
        Encoder:padding_idx:1
        Encoder:self.embed_tokens:Embedding(15912, 256, padding_idx=1)
        Encoder:self.embed_positions:Embedding(1024, 256, padding_idx=1)
        """

        in_channels = convolutions[0][0]  ###256
        ###print("Encoder:in_channels:"+str(in_channels)) ###256
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) // 2
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size, padding=pad,
                        dropout=dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, tokens, positions):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        print("Encoder: tokens:"+str(tokens.size()))
        ###print("Encoder:len(x):"+str(len(x))+" "+str(len(x[0]))+" "+str(len(x[0][0])))  ###x:B,T,D
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x
        ###print("Encoder:input_embedding:"+str(input_embedding))

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=-1)
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        x = grad_multiply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        print("Encoder x output size:"+str(x.size()))
        print("Encoder y output size:"+str(y.size()))
        return x, y


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super(AttentionLayer, self).__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        
        print("AttentionLayer x output size:"+str(x.size()))
        print("AttentionLayer attn_scores output size:"+str(attn_scores.size()))
        return x, attn_scores


class Decoder(nn.Module):
    """Convolutional decoder"""
    def __init__(self, num_embeddings, embed_dim=512, out_embed_dim=256,
                 max_positions=1024, convolutions=((512, 3),) * 20,
                 attention=True, dropout=0.1, padding_idx=1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            pad = kernel_size - 1
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=pad, dropout=dropout))
            self.attention.append(AttentionLayer(out_channels, embed_dim)
                                  if attention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

        self._is_inference_incremental = False

    def forward(self, tokens, positions, encoder_out):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = conv.remove_future_timesteps(x)
            x = F.glu(x)

            # attention
            if attention is not None:
                x = x.transpose(1, 0)
                x, _ = attention(x, target_embedding, (encoder_a, encoder_b))
                x = x.transpose(1, 0)

            # residual
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
 
        print("Decoder x output size:"+str(x.size()))
        return x

    def context_size(self):
        """Maximum number of input elements each output element depends on"""
        context = 1
        for conv in self.convolutions:
            context += conv.kernel_size[0] - 1
        return context

    def incremental_inference(self):
        """Context manager for incremental inference.

        This provides an optimized forward pass for incremental inference
        (i.e., it predicts one time step at a time). If the input order changes
        between time steps, call model.decoder.reorder_incremental_state to
        update the relevant buffers. To generate a fresh sequence, first call
        model.decoder.clear_incremental_state.

        Usage:
        ```
        with model.decoder.incremental_inference():
            for step in range(maxlen):
                out = model.decoder(tokens[:, :step], positions[:, :step],
                                    encoder_out)
                probs = F.log_softmax(out[:, -1, :])
        ```
        """
        class IncrementalInference(object):

            def __init__(self, decoder):
                self.decoder = decoder

            def __enter__(self):
                self.decoder._start_incremental_inference()

            def __exit__(self, *args):
                self.decoder._stop_incremental_inference()

        return IncrementalInference(self)

    def _start_incremental_inference(self):
        assert not self._is_inference_incremental, \
            'already performing incremental inference'
        self._is_inference_incremental = True

        # save original forward and convolution layers
        self._orig_forward = self.forward
        self._orig_conv = self.convolutions

        # switch to incremental forward
        self.forward = self._incremental_forward

        # start a fresh sequence
        self.clear_incremental_state()

    def _stop_incremental_inference(self):
        # restore original forward and convolution layers
        self.forward = self._orig_forward
        self.convolutions = self._orig_conv

        self._is_inference_incremental = False

    def _incremental_forward(self, tokens, positions, encoder_out):
        assert self._is_inference_incremental

        # setup initial state
        if self.prev_state is None:
            # transpose encoder output once to speed up attention layers
            encoder_a, encoder_b = encoder_out
            encoder_a = encoder_a.transpose(1, 2).contiguous()
            self.prev_state = {
                'encoder_out': (encoder_a, encoder_b),
            }

        # load previous state
        encoder_a, encoder_b = self.prev_state['encoder_out']

        # keep only the last token for incremental forward pass
        tokens = tokens[:, -1:]
        positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)
            x = conv.incremental_forward(x)
            x = F.glu(x)

            # attention
            if attention is not None:
                x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b))
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores += attn_scores

            # residual
            x = (x + residual) * math.sqrt(0.5)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = self.fc3(x)

        return x, avg_attn_scores

    def clear_incremental_state(self):
        """Clear all state used for incremental generation.

        **For incremental inference only**

        This should be called before generating a fresh sequence.
        """
        if self._is_inference_incremental:
            self.prev_state = None
            for conv in self.convolutions:
                conv.clear_buffer()

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation).

        **For incremental inference only**

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the choice of beams.
        """
        if self._is_inference_incremental:
            for conv in self.convolutions:
                conv.reorder_buffer(new_order)

    def _use_beamable_mm(self, beam_size):
        """Replace torch.bmm with BeamableMM in attention layers."""
        beamable_mm = BeamableMM(beam_size)
        for attn in self.attention:
            attn.bmm = beamable_mm


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


def grad_multiply(x, scale):
    return GradMultiply.apply(x, scale)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def get_archs():
    return [
        'fconv', 'fconv_giga', 'fconv_iwslt_de_en', 'fconv_wmt_en_ro', 'fconv_wmt_en_de', 'fconv_wmt_en_fr',
    ]


def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown fconv model architecture: {}'.format(args.arch))
    if args.arch != 'fconv':
        # check that architecture is not ambiguous
        for a in ['encoder_embed_dim', 'encoder_layers', 'decoder_embed_dim', 'decoder_layers',
                  'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError('--{} cannot be combined with --arch={}'.format(a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'fconv_iwslt_de_en':
        args.encoder_embed_dim = 256
        args.encoder_layers = '[(256, 3)] * 4'
        args.decoder_embed_dim = 256
        args.decoder_layers = '[(256, 3)] * 3'
        args.decoder_out_embed_dim = 256
    elif args.arch == 'fconv_giga':
        args.encoder_embed_dim = 256
        args.encoder_layers = '[(256, 3)] * 6'
        args.decoder_embed_dim = 256
        args.decoder_layers = '[(256, 3)] * 6'
        args.decoder_out_embed_dim = 256 
    elif args.arch == 'fconv_wmt_en_ro':
        args.encoder_embed_dim = 512
        args.encoder_layers = '[(512, 3)] * 20'
        args.decoder_embed_dim = 512
        args.decoder_layers = '[(512, 3)] * 20'
        args.decoder_out_embed_dim = 512
    elif args.arch == 'fconv_wmt_en_de':
        convs = '[(512, 3)] * 9'       # first 9 layers have 512 units
        convs += ' + [(1024, 3)] * 4'  # next 4 layers have 1024 units
        convs += ' + [(2048, 1)] * 2'  # final 2 layers use 1x1 convolutions
        args.encoder_embed_dim = 768
        args.encoder_layers = convs
        args.decoder_embed_dim = 768
        args.decoder_layers = convs
        args.decoder_out_embed_dim = 512
    elif args.arch == 'fconv_wmt_en_fr':
        convs = '[(512, 3)] * 6'       # first 6 layers have 512 units
        convs += ' + [(768, 3)] * 4'   # next 4 layers have 768 units
        convs += ' + [(1024, 3)] * 3'  # next 3 layers have 1024 units
        convs += ' + [(2048, 1)] * 1'  # next 1 layer uses 1x1 convolutions
        convs += ' + [(4096, 1)] * 1'  # final 1 layer uses 1x1 convolutions
        args.encoder_embed_dim = 768
        args.encoder_layers = convs
        args.decoder_embed_dim = 768
        args.decoder_layers = convs
        args.decoder_out_embed_dim = 512
    else:
        assert args.arch == 'fconv'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    return args

def random_list_generate(min_value,max_value,list_size):  ######generate random numbers which mean 0, stddev 0.1
    rarray=np.random.uniform(min_value,max_value,size=list_size)
    mean=np.average(rarray)
    stddev=np.std(rarray)*10
    return [(value-mean)/stddev for value in list(rarray)]

def build_model(args, dataset):
    ################calculate topic words and build target vocab topic words ids index 
    filename_topic_model = "giga_lda_model0716_"   
    words=[]
    features=[]
    emb_size=0
    topic_word_num=200
    ###f = open(self.params["topic_model.path"],"r")
    f = open(filename_topic_model,"r")
    texts = f.readlines()
    for line in texts: 
       emb_size=len(line.split('\t')[1].split(' '))
       words.append(line.split('\t')[0])
       features.append([float(probability) for probability in line.split('\t')[1].split(' ')[0:emb_size]])
    f.close()    
    samples_size = len(words)    
    topic_words=[]
    for i in range(0,emb_size):
       pro_dict={}
       for j in range(0,samples_size):
            pro_dict[words[j]]=features[j][i]
       prob_list = sorted(pro_dict.items(),key=lambda d:d[1],reverse=True)
       topic_words = topic_words + [item[0] for item in prob_list[0:topic_word_num]]
    topic_words = sorted(list(set(topic_words)))
    
    ### Load topic into memory
    with open(filename_topic_model) as file:
        vocab_topic = list(line.strip("\n") for line in file)
    vocab_topic_size = len(vocab_topic)
    print("vocab_topic_size:"+str(vocab_topic_size))
        
    vocab_topic, topic_embedding = zip(*[_.split("\t") for _ in vocab_topic])
    ###vocab_topic, topic_embedding = zip(*[ [_.split(" ")[0], ' '.join(_.split(" ")[1:257])] for _ in vocab_topic])
    topic_embedding = [list( float(_) for _ in _.split(" ") ) for _ in topic_embedding]
      
    topic_embedding=np.array(topic_embedding)
    topic_embedding[topic_embedding>1.0]=1.0
    topic_embedding=topic_embedding.tolist()
    
    topicword_embedding_dict = dict(zip(vocab_topic,topic_embedding))
      
    topic_emb_size = len(topic_embedding[0])
    ###print("topic_emb_size:"+str(topic_emb_size))
    
    ##vacab_topic_dict = []
    src_dict_word_idx = dataset.src_dict.indices
    vocab_topic_emb = [[float(0)]*topic_emb_size]*len(src_dict_word_idx)
    for word in src_dict_word_idx.keys():
        if word in vocab_topic:
            ##vacab_topic_dict.append(topic_embedding[vocab_topic.index(vocab[vocab_idx])])
            vocab_topic_emb[src_dict_word_idx[word]] = topicword_embedding_dict[word]
        else:
            ##vacab_topic_dict.append( [float(0)]*topic_emb_size )
            ### vocab_topic_emb[vocab_idx] = [float(0)]*topic_emb_size
            vocab_topic_emb[src_dict_word_idx[word]] = random_list_generate(0,1,256)
    
    padding_idx = dataset.dst_dict.pad()
    encoder = Encoder(
        len(dataset.src_dict),
        embed_dim=args.encoder_embed_dim,
        convolutions=eval(args.encoder_layers),
        dropout=args.dropout,
        padding_idx=padding_idx,
        max_positions=args.max_positions,
        vocab_topic_emb=torch.from_numpy(np.array(vocab_topic_emb)),
    )
    decoder = Decoder(
        len(dataset.dst_dict),
        embed_dim=args.decoder_embed_dim,
        convolutions=eval(args.decoder_layers),
        out_embed_dim=args.decoder_out_embed_dim,
        attention=eval(args.decoder_attention),
        dropout=args.dropout,
        padding_idx=padding_idx,
        max_positions=args.max_positions,
    )
    return FConvModel(encoder, decoder, padding_idx)
