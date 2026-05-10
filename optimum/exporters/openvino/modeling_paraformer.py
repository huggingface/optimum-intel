import torch
from torch import nn
import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
import types
import math
import os
import json
import copy
from omegaconf import OmegaConf, DictConfig, ListConfig

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/utils/repeat.py#L14 (Apache 2.0)
class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.

        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).

        """
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args

def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)

# https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/positionwise_feed_forward.py#L14 (Apache 2.0)
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/embedding.py#L416 (Apache 2.0)
class StreamSinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/embedding.py#L383 (Apache 2.0)
class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/embedding.py#L36 (Apache 2.0)
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/layer_norm.py#L13 (Apache 2.0)
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

class BaseTransformerDecoder(nn.Module):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        # Must set by the inheritance
        self.decoders = None

class sequence_mask(nn.Module):
    def __init__(self, max_seq_len=512, flip=True):
        super().__init__()

    def forward(self, lengths, max_seq_len=None, dtype=torch.float32, device=None):
        if max_seq_len is None:
            max_seq_len = lengths.max()
        row_vector = torch.arange(0, max_seq_len, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/multihead_att.py#L67
def preprocess_for_attn(x, mask, cache, pad_fn, kernel_size):
    x = x * mask
    x = x.transpose(1, 2)
    if cache is None:
        x = pad_fn(x)
    else:
        x = torch.cat((cache, x), dim=2)
        cache = x[:, :, -(kernel_size - 1) :]
    return x, cache


# torch_version = tuple([int(i) for i in torch.__version__.split(".")[:2]])
# if torch_version >= (1, 8):
#     import torch.fx

#     torch.fx.wrap("preprocess_for_attn")

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L140 (Apache 2.0)
class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)
        if lora_list is not None:
            if "o" in lora_list:
                self.linear_out = lora.Linear(
                    n_feat, n_feat, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
            else:
                self.linear_out = nn.Linear(n_feat, n_feat)
            lora_qkv_list = ["q" in lora_list, "k" in lora_list, "v" in lora_list]
            if lora_qkv_list == [False, False, False]:
                self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
            else:
                self.linear_q_k_v = lora.MergedLinear(
                    in_feat,
                    n_feat * 3,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=lora_qkv_list,
                )
        else:
            self.linear_out = nn.Linear(n_feat, n_feat)
            self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L353 (Apache 2.0)
class MultiHeadedAttentionSANMExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_out = model.linear_out
        self.linear_q_k_v = model.linear_q_k_v
        self.fsmn_block = model.fsmn_block
        self.pad_fn = model.pad_fn

        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, x, mask):
        mask_3d_btd, mask_4d_bhlt = mask
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask_3d_btd)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask_4d_bhlt)
        return att_outs + fsmn_memory

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, x):
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = self.transpose_for_scores(q)
        k_h = self.transpose_for_scores(k)
        v_h = self.transpose_for_scores(v)
        return q_h, k_h, v_h, v

    def forward_fsmn(self, inputs, mask):
        # b, t, d = inputs.size()
        # mask = torch.reshape(mask, (b, -1, 1))
        inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        x = x * mask
        return x

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L471 (Apache 2.0)
class MultiHeadedAttentionSANMDecoder(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(self, inputs, mask, cache=None, mask_shfit_chunk=None):
        """
        :param x: (#batch, time1, size).
        :param mask: Mask tensor (#batch, 1, time)
        :return:
        """
        # print("in fsmn, inputs", inputs.size())
        b, t, d = inputs.size()
        # logging.info(
        #     "mask: {}".format(mask.size()))
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            # logging.info("in fsmn, mask: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            if mask_shfit_chunk is not None:
                # logging.info("in fsmn, mask_fsmn: {}, {}".format(mask_shfit_chunk.size(), mask_shfit_chunk[0:100:50, :, :]))
                mask = mask * mask_shfit_chunk
            # logging.info("in fsmn, mask_after_fsmn: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            # print("in fsmn, mask", mask.size())
            # print("in fsmn, inputs", inputs.size())
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        b, d, t = x.size()
        if cache is None:
            # print("in fsmn, cache is None, x", x.size())

            x = self.pad_fn(x)
            if not self.training:
                cache = x
        else:
            # print("in fsmn, cache is not None, x", x.size())
            # x = torch.cat((x, cache), dim=2)[:, :, :-1]
            # if t < self.kernel_size:
            #     x = self.pad_fn(x)
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            x = x[:, :, -(self.kernel_size + t - 1) :]
            # print("in fsmn, cache is not None, x_cat", x.size())
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        # print("in fsmn, fsmn_out", x.size())
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L550 (Apache 2.0)
class MultiHeadedAttentionSANMDecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fsmn_block = model.fsmn_block
        self.pad_fn = model.pad_fn
        self.kernel_size = model.kernel_size
        self.attn = None

    def forward(self, inputs, mask, cache=None):
        x, cache = preprocess_for_attn(inputs, mask, cache, self.pad_fn, self.kernel_size)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)

        x = x + inputs
        x = x * mask
        return x, cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L568 (Apache 2.0)
class MultiHeadedAttentionCrossAtt(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        encoder_output_size=None,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        if lora_list is not None:
            if "q" in lora_list:
                self.linear_q = lora.Linear(
                    n_feat, n_feat, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
            else:
                self.linear_q = nn.Linear(n_feat, n_feat)
            lora_kv_list = ["k" in lora_list, "v" in lora_list]
            if lora_kv_list == [False, False]:
                self.linear_k_v = nn.Linear(
                    n_feat if encoder_output_size is None else encoder_output_size, n_feat * 2
                )
            else:
                self.linear_k_v = lora.MergedLinear(
                    n_feat if encoder_output_size is None else encoder_output_size,
                    n_feat * 2,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=lora_kv_list,
                )
            if "o" in lora_list:
                self.linear_out = lora.Linear(
                    n_feat, n_feat, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
            else:
                self.linear_out = nn.Linear(n_feat, n_feat)
        else:
            self.linear_q = nn.Linear(n_feat, n_feat)
            self.linear_k_v = nn.Linear(
                n_feat if encoder_output_size is None else encoder_output_size, n_feat * 2
            )
            self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, x, memory):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """

        # print("in forward_qkv, x", x.size())
        b = x.size(0)
        q = self.linear_q(x)
        q_h = torch.reshape(q, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h

    def forward_attention(self, value, scores, mask, ret_attn=False):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # logging.info(
            #     "scores: {}, mask_size: {}".format(scores.size(), mask.size()))
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        if ret_attn:
            return self.linear_out(x), attn  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, memory, memory_mask, ret_attn=False):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, memory_mask, ret_attn=ret_attn)

    def forward_chunk(self, x, memory, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        if chunk_size is not None and look_back > 0:
            if cache is not None:
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)
                cache["k"] = k_h[:, :, -(look_back * chunk_size[1]) :, :]
                cache["v"] = v_h[:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, -(look_back * chunk_size[1]) :, :],
                    "v": v_h[:, :, -(look_back * chunk_size[1]) :, :],
                }
                cache = cache_tmp
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, None), cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/attention.py#L751 (Apache 2.0)
class MultiHeadedAttentionCrossAttExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_q = model.linear_q
        self.linear_k_v = model.linear_k_v
        self.linear_out = model.linear_out
        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, x, memory, memory_mask, ret_attn=False):
        q, k, v = self.forward_qkv(x, memory)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, memory_mask, ret_attn)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, x, memory):
        q = self.linear_q(x)

        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        return q, k, v

    def forward_attention(self, value, scores, mask, ret_attn):
        scores = scores + mask.to(scores.device)

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        if ret_attn:
            return self.linear_out(context_layer), attn
        return self.linear_out(context_layer)  # (batch, time1, d_model)

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/encoder.py#L44 (MIT License)
class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/encoder.py#L188 (MIT License)
class SANMEncoder(nn.Module):
    """
    Author: Zhifu Gao, Shiliang Zhang, Ming Lei, Ian McLoughlin
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        lora_list: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        selfattention_layer_type: str = "sanm",
        tf2torch_tensor_name_prefix_torch: str = "encoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/encoder",
    ):
        super().__init__()
        self._output_size = output_size
        # input_layer is now force to set to "pe"
        self.embed = SinusoidalPositionEncoder()
        self.normalize_before = normalize_before

        # positionwise_layer_type is now force to set to "linear"
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        # selfattention_layer_type is now force to set to "sanm"
        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
            lora_list,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
            lora_list,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )

        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx

        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.dropout = nn.Dropout(dropout_rate)
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self._output_size

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/encoder.py#L487 (MIT License)
class EncoderLayerSANMExport(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.in_size = model.in_size
        self.size = model.size

    def forward(self, x, mask):

        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x, mask

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/encoder.py#L518 (MIT License)
class SANMEncoderExport(nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        model_name="encoder",
        onnx: bool = True,
        ctc_linear: nn.Module = None,
    ):
        super().__init__()
        self.embed = model.embed
        if isinstance(self.embed, StreamSinusoidalPositionEncoder):
            self.embed = None
        self.model = model
        self.feats_dim = feats_dim
        self._output_size = model._output_size

        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        # from export_model_hf.sanm.attention import MultiHeadedAttentionSANMExport

        if hasattr(model, "encoders0"):
            for i, d in enumerate(self.model.encoders0):
                if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                    d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
                self.model.encoders0[i] = EncoderLayerSANMExport(d)

        for i, d in enumerate(self.model.encoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
            self.model.encoders[i] = EncoderLayerSANMExport(d)

        self.model_name = model_name
        self.num_heads = model.encoders[0].self_attn.h
        self.hidden_size = model.encoders[0].self_attn.linear_out.out_features

        self.ctc_linear = ctc_linear

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, online: bool = False):
        if not online:
            speech = speech * self._output_size**0.5

        mask = self.make_pad_mask(speech_lengths)
        mask = self.prepare_mask(mask)
        if self.embed is None:
            xs_pad = speech
        else:
            xs_pad = self.embed(speech)

        encoder_outs = self.model.encoders0(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        encoder_outs = self.model.encoders(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.model.after_norm(xs_pad)

        if self.ctc_linear is not None:
            xs_pad = self.ctc_linear(xs_pad)
            xs_pad = F.softmax(xs_pad, dim=2)

        return xs_pad, speech_lengths

#Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/sanm/positionwise_feed_forward.py#L12
class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, adim=None, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim if adim is None else adim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = LayerNorm(hidden_units)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/decoder.py#L26 (MIT License)
class DecoderLayerSANM(torch.nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayerSANM, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = torch.nn.Linear(size + size, size)
            self.concat_linear2 = torch.nn.Linear(size + size, size)
        self.reserve_attn = False
        self.attn_mat = []

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/decoder.py#L225 (MIT License)
class ParaformerSANMDecoder(BaseTransformerDecoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        wo_input_layer: bool = False,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        att_layer_num: int = 6,
        kernel_size: int = 21,
        sanm_shfit: int = 0,
        lora_list: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        chunk_multiply_factor: tuple = (1,),
        tf2torch_tensor_name_prefix_torch: str = "decoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/decoder",
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size

        # wo_input_layer is now force to set to False
        # input_layer is now force to set to "embed"
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, attention_dim),
        )

        self.normalize_before = normalize_before

        # self.normalize_before is now force to set to True
        self.after_norm = LayerNorm(attention_dim)
        # use_output_layer is now force to set to True
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks

        self.decoders = repeat(
            att_layer_num,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    lora_list,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        # num_blocks - att_layer_num return 0
        self.decoders2 = None

        self.decoders3 = repeat(
            1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                None,
                None,
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.chunk_multiply_factor = chunk_multiply_factor

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/decoder.py#L592 (MIT License)
class DecoderLayerSANMExport(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.self_attn = model.self_attn
        self.src_attn = model.src_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2 if hasattr(model, "norm2") else None
        self.norm3 = model.norm3 if hasattr(model, "norm3") else None
        self.size = model.size

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):

        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + x

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x = residual + self.src_attn(x, memory, memory_mask)

        return x, tgt_mask, memory, memory_mask, cache

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/decoder.py#L641 (MIT License)
class ParaformerSANMDecoderExport(torch.nn.Module):
    def __init__(self, model, max_seq_len=512, model_name="decoder", onnx: bool = True, **kwargs):
        super().__init__()

        self.model = model

        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        for i, d in enumerate(self.model.decoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                d.self_attn = MultiHeadedAttentionSANMDecoderExport(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttentionCrossAtt):
                d.src_attn = MultiHeadedAttentionCrossAttExport(d.src_attn)
            self.model.decoders[i] = DecoderLayerSANMExport(d)

        if self.model.decoders2 is not None:
            for i, d in enumerate(self.model.decoders2):
                if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                    d.self_attn = MultiHeadedAttentionSANMDecoderExport(d.self_attn)
                self.model.decoders2[i] = DecoderLayerSANMExport(d)

        for i, d in enumerate(self.model.decoders3):
            self.model.decoders3[i] = DecoderLayerSANMExport(d)

        self.output_layer = model.output_layer
        self.after_norm = model.after_norm
        self.model_name = model_name

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hidden: bool = False,
        return_both: bool = False,
    ):

        tgt = ys_in_pad
        tgt_mask = self.make_pad_mask(ys_in_lens)
        tgt_mask, _ = self.prepare_mask(tgt_mask)
        # tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = self.make_pad_mask(hlens)
        _, memory_mask = self.prepare_mask(memory_mask)
        # memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders(x, tgt_mask, memory, memory_mask)
        if self.model.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.model.decoders2(
                x, tgt_mask, memory, memory_mask
            )
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(x, tgt_mask, memory, memory_mask)
        hidden = self.after_norm(x)
        # x = self.output_layer(x)

        if self.output_layer is not None and return_hidden is False:
            x = self.output_layer(hidden)
            return x, ys_in_lens
        if return_both:
            x = self.output_layer(hidden)
            return x, hidden, ys_in_lens
        return hidden, ys_in_lens

# Modified from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/export_meta.py#L11 (MIT License)
def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    model.encoder = SANMEncoderExport(model.encoder, onnx=is_onnx)
    model.predictor = CifPredictorV2Export(model.predictor, onnx=is_onnx)
    model.decoder = ParaformerSANMDecoderExport(model.decoder, onnx=is_onnx)
    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    # model.export_name = "model"
    return model


def export_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    # batch = to_device(batch, device=self.device)

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    # sample_ids = decoder_out.argmax(dim=-1)

    return decoder_out, pre_token_length


def export_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_input_names(self):
    return ["speech", "speech_lengths"]


def export_output_names(self):
    return ["logits", "token_num"]


def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "logits": {0: "batch_size", 1: "logits_length"},
        "token_num": {0: "batch_size"}
    }


def export_name(
    self,
):
    return "model"

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/cif_predictor.py#L173 (MIT License)
class CifPredictorV2(torch.nn.Module):
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.0,
        tf2torch_tensor_name_prefix_torch="predictor",
        tf2torch_tensor_name_prefix_tf="seq2seq/cif",
        tail_mask=True,
    ):
        super().__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.tail_mask = tail_mask

# Copied from https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/cif_predictor.py#L431 (MIT License)
class CifPredictorV2Export(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        self.pad = model.pad
        self.cif_conv1d = model.cif_conv1d
        self.cif_output = model.cif_output
        self.threshold = model.threshold
        self.smooth_factor = model.smooth_factor
        self.noise_threshold = model.noise_threshold
        self.tail_threshold = model.tail_threshold

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ):
        alphas, token_num = self.forward_cnn(hidden, mask)
        mask = mask.transpose(-1, -2).float()
        mask = mask.squeeze(-1)
        hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, mask=mask)
        acoustic_embeds, cif_peak = cif_v1_export(hidden, alphas, self.threshold)

        return acoustic_embeds, token_num, alphas, cif_peak

    def forward_cnn(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        mask = mask.transpose(-1, -2).float()
        alphas = alphas * mask
        alphas = alphas.squeeze(-1)
        token_num = alphas.sum(-1)

        return alphas, token_num

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold

        zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
        ones_t = torch.ones_like(zeros_t)

        mask_1 = torch.cat([mask, zeros_t], dim=1)
        mask_2 = torch.cat([ones_t, mask], dim=1)
        mask = mask_2 - mask_1
        tail_threshold = mask * tail_threshold
        alphas = torch.cat([alphas, zeros_t], dim=1)
        alphas = torch.add(alphas, tail_threshold)

        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor


@torch.jit.script
def cif_v1_export(hidden, alphas, threshold: float):
    device = hidden.device
    dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.size()
    threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device)

    frames = torch.zeros(batch_size, len_time, hidden_size, dtype=dtype, device=device)
    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

    # prefix_sum = torch.cumsum(alphas, dim=1)
    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(
        torch.float32
    )  # cumsum precision degradation cause wrong result in extreme
    prefix_sum_floor = torch.floor(prefix_sum)
    dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
    dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)

    dislocation_prefix_sum_floor[:, 0] = 0
    dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor

    fire_idxs = dislocation_diff > 0
    fires[fire_idxs] = 1
    fires = fires + prefix_sum - prefix_sum_floor

    # prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1).tile((1, 1, hidden_size)) * hidden, dim=1)
    prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1).repeat((1, 1, hidden_size)) * hidden, dim=1)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    # remain_frames = remains[fire_idxs].unsqueeze(-1).tile((1, hidden_size)) * hidden[fire_idxs]
    remain_frames = remains[fire_idxs].unsqueeze(-1).repeat((1, hidden_size)) * hidden[fire_idxs]

    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    # max_label_len = batch_len.max()
    max_label_len = alphas.sum(dim=-1)
    max_label_len = torch.floor(max_label_len).max().to(dtype=torch.int64)

    # frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    frame_fires_idxs = indices < batch_len.unsqueeze(1)
    frame_fires[frame_fires_idxs] = frames
    return frame_fires, fires

# https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/model.py#L30 (MIT License)
class Paraformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        predictor: str = None,
        predictor_conf: Optional[Dict] = None,
        ctc_weight: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        # report_cer: bool = True,
        # report_wer: bool = True,
        # sym_space: str = "<space>",
        # sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        # predictor=None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        use_1st_decoder_loss: bool = False,
        **kwargs,
    ):

        super().__init__()
        encoder = SANMEncoder(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if decoder is not None:
            decoder = ParaformerSANMDecoder(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **decoder_conf,
            )

        if predictor is not None:
            predictor = CifPredictorV2(**predictor_conf)

        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

    def export(self, **kwargs):

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models

def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):

    if isinstance(file_path_metas, dict):
        if isinstance(cfg, list):
            cfg.append({})

        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    if isinstance(cfg, dict):
                        cfg[k] = p
                    elif isinstance(cfg, list):
                        # if len(cfg) == 0:
                        # cfg.append({})
                        cfg[-1][k] = p

            elif isinstance(v, dict):
                if isinstance(cfg, dict):
                    if k not in cfg:
                        cfg[k] = {}
                    add_file_root_path(model_or_path, v, cfg[k])
                # elif isinstance(cfg, list):
                #     cfg.append({})
                #     add_file_root_path(model_or_path, v, cfg)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if k not in cfg:
                        cfg[k] = []
                    if isinstance(vv, str):
                        p = os.path.join(model_or_path, vv)
                        # file_path_metas[i] = p
                        if os.path.exists(p):
                            if isinstance(cfg[k], dict):
                                cfg[k] = p
                            elif isinstance(cfg[k], list):
                                cfg[k].append(p)
                    elif isinstance(vv, dict):
                        add_file_root_path(model_or_path, vv, cfg[k])

    return cfg

def get_or_download_model_dir_hf(
    model,
    model_revision=None,
    is_training=False,
    check_latest=True,
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
        model_revision  (str, optional): model version number.
        :param is_training:
    """
    from huggingface_hub import snapshot_download

    model_cache_dir = snapshot_download(model)
    return model_cache_dir

def download_from_hf(**kwargs):
    model_or_path = kwargs.get("model")
    model_revision = kwargs.get("model_revision", "master")
    if not os.path.exists(model_or_path) and "model_path" not in kwargs:
        try:
            model_or_path = get_or_download_model_dir_hf(
                model_or_path,
                model_revision,
                is_training=kwargs.get("is_training"),
                check_latest=kwargs.get("check_latest", True),
                )
        except Exception as e:
            print(f"Download: {model_or_path} failed!: {e}")

    kwargs["model_path"] = model_or_path if "model_path" not in kwargs else kwargs["model_path"]

    if os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8") as f:
            conf_json = json.load(f)
            cfg = {}
            if "file_path_metas" in conf_json:
                add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            cfg.update(kwargs)
            if "config" in cfg:
                config = OmegaConf.load(cfg["config"])
                kwargs = OmegaConf.merge(config, cfg)
                kwargs["model"] = config["model"]
    elif os.path.exists(os.path.join(model_or_path, "config.yaml")):
        config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
        kwargs = OmegaConf.merge(config, kwargs)
        init_param = os.path.join(model_or_path, "model.pt")
        if "init_param" not in kwargs or not os.path.exists(kwargs["init_param"]):
            kwargs["init_param"] = init_param
            assert os.path.exists(kwargs["init_param"]), "init_param does not exist"
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.json")
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(model_or_path, "seg_dict")
        kwargs["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
    if isinstance(kwargs, DictConfig):
        kwargs = OmegaConf.to_container(kwargs, resolve=True)

    return kwargs

def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value

def load_pretrained_model(
    path: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool = True,
    map_location: str = "cpu",
    oss_bucket=None,
    scope_map=[],
    excludes=None,
    **kwargs,
):
    """Load a model state and set it to the model.

    Args:
            init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:

    """

    obj = model
    dst_state = obj.state_dict()
    ori_state = torch.load(path, map_location=map_location)

    src_state = copy.deepcopy(ori_state)
    src_state = src_state["state_dict"] if "state_dict" in src_state else src_state
    src_state = src_state["model_state_dict"] if "model_state_dict" in src_state else src_state
    src_state = src_state["model"] if "model" in src_state else src_state

    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    scope_map += ["module.", "None"]
    logging.info(f"scope_map: {scope_map}")

    for k in dst_state.keys():
        excludes_flag = False
        if excludes is not None:
            for k_ex in excludes:
                if k.startswith(k_ex):
                    logging.info(f"key: {k} matching: {k_ex}, excluded")
                    excludes_flag = True
                    break
        if excludes_flag:
            continue

        k_src = k

        if scope_map is not None:
            src_prefix = ""
            dst_prefix = ""
            for i in range(0, len(scope_map), 2):
                src_prefix = scope_map[i] if scope_map[i].lower() != "none" else ""
                dst_prefix = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""

                if dst_prefix == "" and (src_prefix + k) in src_state.keys():
                    k_src = src_prefix + k
                    if not k_src.startswith("module."):
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")
                elif (
                    k.startswith(dst_prefix)
                    and k.replace(dst_prefix, src_prefix, 1) in src_state.keys()
                ):
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                logging.info(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]
        else:
            print(f"Warning, miss key in ckpt: {k}, {path}")

    flag = obj.load_state_dict(dst_state, strict=True)

def _torchscripts(model, path, device="cuda"):
    dummy_input = model.export_dummy_inputs()
    model_jit_script = torch.jit.trace(model, dummy_input)
    return model_jit_script

def export_utils(
    model, data_in=None, quantize: bool = False, opset_version: int = 14, type="onnx", **kwargs
):
    model_scripts = model.export(**kwargs)
    export_dir = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))
    os.makedirs(export_dir, exist_ok=True)

    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)
    for m in model_scripts:
        m.eval()
        device = "cpu"    
        print("Exporting torchscripts on device {}".format(device))
        model_jit_scripts = _torchscripts(m, path=export_dir, device=device)

    return export_dir, model_jit_scripts

def download_model(**kwargs):
    kwargs = download_from_hf(**kwargs)
    return kwargs

def build_model(**kwargs):
    assert "model" in kwargs
    kwargs = download_model(**kwargs)
    torch.set_num_threads(kwargs.get("ncpu", 4))

    # build tokenizer
    # Here to remove building tokenizer to get vocab_size. Currently hard_code the value here
    # Check the downloaded token.json and the vocab_size is the token number in token.json
    kwargs["vocab_size"] = 8404

    # build model
    model_conf = {}
    deep_update(model_conf, kwargs.get("model_conf", {}))
    deep_update(model_conf, kwargs)
    model = Paraformer(**model_conf)

    # init_param
    init_param = kwargs.get("init_param", None)
    if init_param is not None:
        if os.path.exists(init_param):
            logging.info(f"Loading pretrained params from {init_param}")
            load_pretrained_model(
                model=model,
                path=init_param,
                ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                oss_bucket=kwargs.get("oss_bucket", None),
                scope_map=kwargs.get("scope_map", []),
                excludes=kwargs.get("excludes", None),
            )
        else:
            print(f"error, init_param does not exist!: {init_param}")

    # fp16
    if kwargs.get("fp16", False):
        model.to(torch.float16)
    elif kwargs.get("bf16", False):
        model.to(torch.bfloat16)
    model.to(kwargs["device"])

    return model, kwargs

def export(model, kwargs, input=None, **cfg):
    del kwargs["model"]
    model.eval()

    with torch.no_grad():
        export_dir, model_jit_scripts = export_utils(model=model, **kwargs)

    return export_dir, model_jit_scripts
