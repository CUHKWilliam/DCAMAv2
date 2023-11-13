import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
        x = attention_group_keys(query, key, value, mask=mask, dropout=self.dropout, bank_size=256)

        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

# from pytorch3d.ops import sample_farthest_points
# def attention_group_keys(query, key, value, mask=None, dropout=None, bank_size=2048, training=True):
#     # "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     batch_size, num_heads, num_v = query.size(0), query.size(1), value.size(2)
#     num_k = key.size(2)
#     attn_val = []
#     losses = torch.tensor(0.).float().cuda()
#     for b in range(batch_size):
#         attn_val_b = []
#         for head_idx in range(num_heads):
#             key_h = key[b, head_idx, ...]
#             query_h, val_h = query[b, head_idx, ...], value[b, head_idx, ...]
#             key_h2, indices = sample_farthest_points(key_h.unsqueeze(0), K=torch.tensor([bank_size]).int().cuda())
#             key_h2, indices = key_h2.squeeze(0), indices.squeeze(0)
#             val_h2 = val_h[indices]
#             attn_val_h = (F.softmax(torch.matmul(query_h, key_h2.transpose(-2, -1)) / math.sqrt(d_k), dim=-1) * val_h2.permute(1, 0)).sum(-1)
#             attn_val_b.append(attn_val_h)
#         attn_val_b = torch.stack(attn_val_b, dim=0)
#         attn_val.append(attn_val_b)
#     attn_val = torch.stack(attn_val, dim=0).unsqueeze(-1)
#     return attn_val


from libKMCUDA import kmeans_cuda
def attention_group_keys(query, key, value, mask=None, dropout=None, bank_size=2048):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    batch_size, num_heads, num_v = query.size(0), query.size(1), value.size(2)
    num_k = key.size(2)
    attn_val = []
    for b in range(batch_size):
        attn_val_b = []
        for head_idx in range(num_heads):
            key_h = key[b, head_idx, ...]
            query_h, val_h = query[b, head_idx, ...], value[b, head_idx, ...]
            current_device = torch.cuda.current_device()
            centroids, assignments = kmeans_cuda(key_h.detach().cpu().numpy().astype(np.float16), bank_size, init='random', verbosity=0)
            torch.cuda.set_device(current_device)
            nan_indices = np.where(np.isnan(centroids).any(-1))[0]
            mask = torch.from_numpy(np.logical_not(np.in1d(assignments, nan_indices))).bool().cuda()
            assignments = torch.from_numpy(assignments.astype(np.int32)).long().cuda()
            assignments, key_h, val_h = assignments[mask], key_h[mask], val_h[mask]
            cluster_ids_x = assignments
            cluster_centers = centroids# [np.logical_not(np.isnan(centroids).any(-1))]
            cluster_centers = torch.from_numpy(cluster_centers).float().cuda()
            val_h2 = torch.zeros(bank_size, 1).float().cuda()
            for ind in range(bank_size):
                val_h2[ind] = torch.nan_to_num(torch.mean(val_h[cluster_ids_x == ind], dim=0), nan=0.)
            key_h2 = cluster_centers.cuda()
            attn_val_h = (F.softmax(torch.matmul(query_h, key_h2.transpose(-2, -1)) / math.sqrt(d_k), dim=-1) * val_h2.permute(1, 0)).sum(-1)
            attn_val_b.append(attn_val_h)
        attn_val_b = torch.stack(attn_val_b, dim=0)
        attn_val.append(attn_val_b)
    attn_val = torch.stack(attn_val, dim=0).unsqueeze(-1)
    return attn_val


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) \
#              / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
