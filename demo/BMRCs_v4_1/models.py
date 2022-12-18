import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import copy


class H_TransformerEncoder(nn.Module):

    '''
    Transformer Encoder
    '''

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 inner_encoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5):

        super(H_TransformerEncoder, self).__init__()
        encoder_layer = H_TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  inner_layer=inner_encoder_layers,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  activation=activation,
                                                  layer_norm_eps=layer_norm_eps)

        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.num_layers = num_encoder_layers
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        '''
        Initiate parameters in the transformer model
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, src_key_padding_mask):  # [表示相同操作重复num_encoder_layer次]
        # src:[bs,S,E]
        B, L, _ = src.shape
        output = src.transpose(0, 1)  # src:[S,bs,E]

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.encoder_norm is not None:
            output = self.encoder_norm(output)

        return output.transpose(0, 1)  # [bs,S,E]


class H_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, inner_layer=3, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super(H_TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        inner_encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                      nhead=nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      layer_norm_eps=layer_norm_eps)
        self.layers = _get_clones(inner_encoder_layer, inner_layer)
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask, src_key_padding_mask, isindi=True):

        output = src
        L, B, D = src.shape

        for idx, layer in enumerate(self.layers):
            # mask需要改成多头的
            mask_indi = src_mask[idx].bool().int() if isindi else ~(src_mask[idx].bool()).int()
            rm_inf = (mask_indi.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, mask_indi.shape[-1])
            attn_mask = mask_indi.float().masked_fill(mask_indi == 0, float('-inf')).masked_fill(mask_indi > 0, float(
                0.0)).masked_fill(rm_inf, float(0.0))
            attn_mask = torch.stack([attn_mask for _ in range(self.nhead)], dim=1).contiguous().view(-1, L, L)

            output = layer(output, src_mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

            assert (torch.isnan(output).sum() == 0)

        if self.encoder_norm is not None:
            output = self.encoder_norm(output)
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.feedforword = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            self.dropout,
            nn.Linear(dim_feedforward, d_model),
            self.dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src_add_norm = self.norm(src + self.dropout(src2))
        return self.norm(src + self.feedforword(src_add_norm))


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.kdim)
        self.v_proj = nn.Linear(embed_dim, self.vdim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        L, B, D = query.size()
        single_attn_mask = attn_mask.contiguous().view(B, -1, L, L)[:, 0, :, :]
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * scaling

        # check attn_mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [B * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(L, B * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == B
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q,
                                        k.transpose(1, 2))  # [B*num_heads,L,D] * [B*num_heads,D,L] -->[B*num_heads,L,L]
        assert list(attn_output_weights.size()) == [B * self.num_heads, L, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B,N,L,L]->[B,1,1,L]
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(B * self.num_heads, L, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)  # [B,N,L,L] [B,N,L,D]
        assert list(attn_output.size()) == [B * self.num_heads, L, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, B, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class RelationAttention(nn.Module):
    def __init__(self, in_dim = 768, hidden_dim = 64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''
    def __init__(self, in_dim = 300, mem_dim = 300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v) # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature) # (N, L, D)
        Q = self.linear(Q) # (N, L, D)
        feature = self.linear(feature) # (N, L, D)

        att_feature = torch.cat([feature, Q], dim = 2) # (N, L, 2D)
        att_weight = self.fc(att_feature) # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out


class DotprodAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = aspect_v
        Q = Q.unsqueeze(2)  # (N, D, 1)
        dot_prod = torch.bmm(feature, Q)  # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, D, 1)
        attention_weight = mask_logits(dot_prod, dmask)  # (N, L ,1)
        attention = F.softmax(attention_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        # (N, D), ([N, L]), (N, L, 1)
        return out

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepparseMultiHeadAttention(nn.Module):
    def __init__(self, h=6, Co=300, cat=True):
        super().__init__()
        self.hidden_size = Co // h
        self.h = h
        self.fc1 = nn.Linear(Co, Co)
        self.relu = nn.ReLU()
        self.fc2s = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(h)])
        self.cat = cat

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        nbatches = dep_tags_v.size(0)
        Q = self.fc1(dep_tags_v).view(nbatches, -1, self.h,
                                      self.hidden_size)  # [N, L, #heads, hidden_size]
        Q = self.relu(Q)
        Q = Q.transpose(0, 2)  # [#heads, L, N, hidden_size]
        Q = [l(q).squeeze(2).transpose(0, 1)
             for l, q in zip(self.fc2s, Q)]  # [N, L] * #heads
        # Q = Q.squeeze(2)
        Q = [F.softmax(mask_logits(q, dmask), dim=1).unsqueeze(2)
             for q in Q]  # [N, L, 1] * #heads

        # Q = Q.unsqueeze(2)
        if self.cat:
            out = torch.cat(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=1)
        else:
            out = torch.stack(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=2)
            out = torch.sum(out, dim=2)
        # out = out.squeeze(2)
        return out, Q[0]  # ([N, L]) one head