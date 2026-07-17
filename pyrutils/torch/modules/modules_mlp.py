
import copy

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Optional, Sequence, Union
from pyrutils.torch.general import pick_activation_function

#----------------------------------------------------
""" 搭建多层感知机 """
def build_mlp(dims: Sequence[int], activations: Optional[Sequence[Union[str, dict]]] = None,
              dropout: float = 0.0, bias: bool = True, norm: bool = False):
    """Build a general Multi-layer Perceptron (MLP).
    Arguments:
        dims - An iterable containing the sequence of input/hidden/output dimensions. For instance, if
            dims = [256, 128, 64], our MLP receives input features of dimension 256, reduces it to 128, and outputs
            features of dimension 64.       [输入层/隐藏层/输出层]维度
        activations - An iterable containing the activations of each layer of the MLP. Each element of the iterable can
            be either a string or a dictionary. If it is a string, it specifies the name of the activation function,
            such as 'relu'; if it is a dictionary, it should contain a name key, and optional keyword arguments for the
            function. For instance, a valid input could be ['relu', {'name': 'logsoftmax', 'dim': -1}]. If activations
            is None, no activation functions are applied to the outputs of the layers of the MLP.       激活函数
        dropout - Dropout probability.      dropout率
        bias - Whether to include a bias term in the linear layers or not.      是否在线性层加入偏置
    Returns:
        An MLP as a PyTorch Module.
    """
    if activations is None:
        activations = ['identity'] * (len(dims) - 1)
    if len(dims) - 1 != len(activations):
        raise ValueError('Number of activations must be the same as the number of dimensions - 1.')
    layers = []
    for dim_in, dim_out, activation in zip(dims[:-1], dims[1:], activations):
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        layers.append(pick_activation_function(activation))
        if norm:
            layers.append(nn.LayerNorm(dim_out))
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        
    return nn.Sequential(*layers)



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        query = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, src_pos=pos_embed)
        hs = self.decoder(query, memory, memory_key_padding_mask=mask,
                          memory_pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, src_pos=src_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, query, memory,
                query_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None
                ):
        output = query

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, query_mask=query_mask,
                           memory_mask=memory_mask,
                           query_key_padding_mask=query_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=query_pos, memory_pos=memory_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, src_pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, src_pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, src_pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, memory,
                     query_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     query_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     memory_pos: Optional[Tensor] = None
                     ):
        q = k = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, k, value=query, attn_mask=query_mask,
                              key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query

    def forward_pre(self, query, memory,
                    query_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    query_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None
                    ):
        query2 = self.norm1(query)
        q = k = self.with_pos_embed(query2, query_pos)
        query2 = self.self_attn(q, k, value=query2, attn_mask=query_mask,
                              key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout1(query2)
        query2 = self.norm2(query)
        qq = self.with_pos_embed(query2, query_pos)
        kk = self.with_pos_embed(memory, memory_pos)
        value = memory
        query2 = self.multihead_attn(query=qq,
                                   key=kk,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        query = query + self.dropout2(query2)
        query2 = self.norm3(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
        query = query + self.dropout3(query2)
        return query

    def forward(self, query, memory,
                query_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None
                ):
        if self.normalize_before:
            return self.forward_pre(query, memory, query_mask, memory_mask,
                                    query_key_padding_mask, memory_key_padding_mask, query_pos, memory_pos)
        return self.forward_post(query, memory, query_mask, memory_mask,
                                 query_key_padding_mask, memory_key_padding_mask, query_pos, memory_pos)


class CaptionGuidedModule(nn.Module):
    def __init__(self,
                 d_model: int=512,
                 num_layers: int = 2,
                #  alpha: float = 0.1
                #  device = 'cuda'
                 ):
        super().__init__()
        # self.sa = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*4, normalize_before=True)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=decoder_norm)
        # self.alpha = nn.Parameter(torch.ones(d_model) * alpha)
        # self.device = device
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x_h, caption,
                caption_pos: Optional[Tensor] = None,
                caption_mask: Optional[Tensor] = None):
        # _, time_step, num_human, _ = x_h.shape        # bs, t, num_queries, hs

        # generate visual prompts for each human-object combination
        # for i in range(time_step):
        #     single_query_fea = hs_e[:, :, i, :].transpose(0,1)
        totoal_prompt = []
        for i in range(x_h.shape[2]):
            prompt = self.decoder(query=caption,
                                        memory=x_h[:, :, i],
                                        query_key_padding_mask=None,
                                        query_pos=None
                                            )[0]

            totoal_prompt.append(prompt)

        prompt = torch.stack(totoal_prompt, dim=1)

        attention_weight = F.softmax(torch.matmul(prompt.transpose(0, 2), x_h.permute(1, 2, 3, 0)), dim=-1)
        return attention_weight
    


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")