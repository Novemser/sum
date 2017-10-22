# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
import torch.nn.functional as F
from .conv_tbc import ConvTBC


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    This module replaces convolutions with linear layers as appropriate
    and supports optimizations for incremental inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.clear_buffer()

        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def remove_future_timesteps(self, x):
        """Remove future time steps created by padding."""
        if self.kernel_size[0] > 1 and self.padding[0] > 0:
            x = x[:-self.padding[0], :, :]
        return x

    def incremental_forward(self, input, enable_bp=False):
        """Forward convolution one time step at a time.

        This function maintains an internal state to buffer signal and
        accepts a single frame as input. If the input order changes
        between time steps, call reorder_buffer. To apply to fresh
        inputs, call clear_buffer.
        """
        if self.training:
            raise RuntimeError('LinearizedConvolution only supports inference')

        # run forward pre hooks (e.g., weight norm)
        # enable_bp???
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        kw = self.kernel_size[0]
        bsz = input.size(0)  # input: bsz x len x dim
        if not enable_bp:
            # reshape weight
            weight = self._get_linearized_weight()
            if kw > 1:
                input = input.data
                if self.input_buffer is None:
                    self.input_buffer = input.new(bsz, kw, input.size(2))
                    self.input_buffer.zero_()
                else:
                    # shift buffer
                    self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
                # append next input
                self.input_buffer[:, -1, :] = input[:, -1, :]
                input = torch.autograd.Variable(self.input_buffer, volatile=True)
            output = F.linear(input.view(bsz, -1), weight, self.bias)
            return output.view(bsz, 1, -1)
        else:
            # reshape weight
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            weight = weight.view(self.out_channels, -1)
            
            if kw > 1:
                if self.input_buffer is None:
                    self.input_buffer = input.data.new(bsz, kw, input.size(2))
                    self.input_buffer.zero_()
                    self.input_buffer = torch.autograd.Variable(self.input_buffer)
                else:
                    # shift buffer
                    _next_buffer = self.input_buffer[:, 1:, :].clone()
                    self.input_buffer[:, :-1, :] = _next_buffer
                # append next input
                self.input_buffer[:, -1, :] = input[:, -1, :].clone()
            output = F.linear(self.input_buffer.view(bsz, -1), weight, self.bias)
            return output.view(bsz, 1, -1)
            '''
            self.input_buffer = self.input_buffer.transpose(0, 1) # kw * bsz * dim
            output = self.forward(self.input_buffer)
            output = self.remove_future_timesteps(output)
            output = output.view(bsz, -1)
            self.input_buffer = self.input_buffer.transpose(0, 1)
            '''
            ## TODO: use ordinary forward


    def clear_buffer(self):
        self.input_buffer = None

    def reorder_buffer(self, new_order, enable_bp=False):
        if self.input_buffer is not None:
            if not enable_bp:
                self.input_buffer = self.input_buffer.index_select(0, new_order)
            else:
                self.input_buffer = torch.index_select(self.input_buffer, 
                                                       0, 
                                                       torch.autograd.Variable(new_order, requires_grad=False))


    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None
