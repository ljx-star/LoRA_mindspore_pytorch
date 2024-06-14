import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, Zero, HeUniform, initializer

import math
from typing import Optional, List


class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(keep_prob=1 - lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    def __init__(self, vocab_size: int, embedding_size: int, r: int = 0, lora_alpha: int = 1,
                 merge_weights: bool = True, **kwargs):
        nn.Embedding.__init__(self, vocab_size=vocab_size, embedding_size=embedding_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = Parameter(Tensor(shape=(r, vocab_size), dtype=mindspore.float32, init=Zero()))
            self.lora_B = Parameter(Tensor(shape=(embedding_size, r), dtype=mindspore.float32, init=Normal(0.02)))
            self.scaling = self.lora_alpha / self.r
            self.embedding_table.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        super(Embedding, self).init_parameters_data()
        if hasattr(self, 'lora_A'):
            self.lora_A.set_data(Tensor(self.lora_A.shape, dtype=mindspore.float32, init=Zero()))
            self.lora_B.set_data(Tensor(self.lora_B.shape, dtype=mindspore.float32, init=Normal(0.02)))

    def set_train(self, mode: bool = True):
        super(Embedding, self).set_train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.embedding_table.set_data(
                        self.embedding_table.data - ops.matmul(self.lora_B, self.lora_A).transpose() * self.scaling)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.embedding_table.set_data(
                        self.embedding_table.data + ops.matmul(self.lora_B, self.lora_A).transpose() * self.scaling)
                self.merged = True

    def construct(self, x: Tensor):
        if self.r > 0 and not self.merged:
            result = super(Embedding, self).construct(x)
            after_A = ops.gather(self.lora_A.transpose(), x, 0)
            result += ops.matmul(after_A, self.lora_B.transpose()) * self.scaling
            return result
        else:
            return super(Embedding, self).construct(x)


class Linear(nn.Dense, LoRALayer):
    def __init__(self, in_channels: int, out_channels: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.,
                 fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs):
        nn.Dense.__init__(self, in_channels, out_channels, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = Parameter(
                Tensor(shape=(r, in_channels), dtype=mindspore.float32, init=HeUniform(math.sqrt(5))))
            self.lora_B = Parameter(Tensor(shape=(out_channels, r), dtype=mindspore.float32, init=Zero()))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_data(self.weight.data.transpose())

    def reset_parameters(self):
        super(Linear, self).reset_parameters()
        if hasattr(self, 'lora_A'):
            self.lora_A.set_data(Tensor(self.lora_A.shape, dtype=mindspore.float32, init=HeUniform(math.sqrt(5))))
            self.lora_B.set_data(Tensor(self.lora_B.shape, dtype=mindspore.float32, init=Zero()))

    def construct(self, x: Tensor):
        def T(w):
            return w.transpose() if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = super(Linear, self).construct(x)
            result += (self.lora_dropout(x) @ self.lora_A.transpose() @ self.lora_B.transpose()) * self.scaling
            return result
        else:
            return super(Linear, self).construct(x)


class MergedLinear(nn.Dense, LoRALayer):
    def __init__(self, in_channels: int, out_channels: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.,
                 enable_lora: list = [False], fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs):
        nn.Dense.__init__(self, in_channels, out_channels, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_channels % len(enable_lora) == 0, 'The length of enable_lora must divide out_channels'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = Parameter(Tensor(shape=(r * sum(enable_lora), in_channels), dtype=mindspore.float32,
                                           init=HeUniform(math.sqrt(5))))
            self.lora_B = Parameter(
                Tensor(shape=(out_channels // len(enable_lora) * sum(enable_lora), r), dtype=mindspore.float32,
                       init=Zero()))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = mnp.zeros((out_channels,), dtype=mindspore.bool_)
            self.lora_ind[mnp.array(self.enable_lora)] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.set_data(self.weight.data.transpose())

    def reset_parameters(self):
        super(MergedLinear, self).reset_parameters()
        if hasattr(self, 'lora_A'):
            self.lora_A.set_data(Tensor(self.lora_A.shape, dtype=mindspore.float32, init=HeUniform(math.sqrt(5))))
            self.lora_B.set_data(Tensor(self.lora_B.shape, dtype=mindspore.float32, init=Zero()))

    def zero_pad(self, x):
        result = ops.Zeros((len(self.lora_ind),) + x.shape[1:], dtype=x.dtype)
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose() if self.fan_in_fan_out else w

        delta_w = ops.Conv1D(self.lora_A.expand_dims(0), self.lora_B.expand_dims(-1),
                             group=sum(self.enable_lora)).squeeze(0)
        return T(self.zero_pad(delta_w))

    def construct(self, x: Tensor):
        def T(w):
            return w.transpose() if self.fan_in_fan_out else w

        if self.merged:
            return super(MergedLinear, self).construct(x)
        else:
            result = super(MergedLinear, self).construct(x)
            if self.r > 0:
                result += ops.matmul(self.lora_dropout(x), T(self.merge_AB().transpose())) * self.scaling
            return result


class ConvLoRA(nn.Cell, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        if r > 0:
            self.lora_A = Parameter(Tensor(shape=(r * kernel_size, in_channels * kernel_size), dtype=mindspore.float32,
                                           init=HeUniform(math.sqrt(5))))
            self.lora_B = Parameter(
                Tensor(shape=(out_channels // self.conv.group * kernel_size, r * kernel_size), dtype=mindspore.float32,
                       init=Zero()))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.init_parameters_data()
        if hasattr(self, 'lora_A'):
            self.lora_A.set_data(Tensor(self.lora_A.shape, dtype=mindspore.float32, init=HeUniform(math.sqrt(5))))
            self.lora_B.set_data(Tensor(self.lora_B.shape, dtype=mindspore.float32, init=Zero()))

    def construct(self, x):
        if self.r > 0 and not self.merged:
            weight = self.conv.weight + (ops.matmul(self.lora_B, self.lora_A)).view(
                self.conv.weight.shape) * self.scaling
            return self.conv.construct(x, weight, self.conv.bias)
        return self.conv.construct(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)