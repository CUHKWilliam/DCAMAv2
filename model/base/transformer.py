import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

TRAIN = False
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

        query_dir, key_dir = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip([self.linears[0], self.linears[0]], (query, key))]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        query_norm = self.linears[1](query)[:, :, :self.h].view(nbatches, -1, self.h).transpose(1, 2)
        key_norm = self.linears[1](key)[:, :, :self.h].view(nbatches, -1, self.h).transpose(1, 2)
        # query_norm[query_norm < 1.] *= 0.
        # key_norm[key_norm < 1.] *= 0.
        if not TRAIN:
            query_dir = query_dir.detach().cpu()
            key_dir = key_dir.detach().cpu()
            query_norm = query_norm.detach().cpu()
            key_norm = key_norm.detach().cpu()
            value = value.cpu()


        query = query_dir / query_dir.norm(dim=-1).unsqueeze(-1) * 10 * query_norm.unsqueeze(-1)
        key = key_dir / key_dir.norm(dim=-1).unsqueeze(-1) * 10 * key_norm.unsqueeze(-1)

        # query = query_dir / query_dir.norm(dim=-1).unsqueeze(-1)
        # key = key_dir / key_dir.norm(dim=-1).unsqueeze(-1)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x = attention_group_keys(query, key, value, mask=mask, dropout=self.dropout, bank_size=2048)
        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 2)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#
#         query, key = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip([self.linears[0], self.linears[1]], (query, key))]
#         value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
#
#         if not TRAIN:
#             query = query.detach().cpu()
#             key = key.detach().cpu()
#             value = value.cpu()
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
#         # x = attention_group_keys(query, key, value, mask=mask, dropout=self.dropout, bank_size=2048)
#         # 3) "Concat" using a view and apply a final linear.
#         return torch.mean(x, -3)

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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def plot_pts(pts, centers):
    pca = PCA(n_components=2)
    num_pts = len(pts)
    pts2 = np.concatenate((pts, centers), axis=0)
    Xt = pca.fit_transform(pts2, )
    plt.scatter(Xt[:num_pts, 0], Xt[:num_pts, 1], c=[0, 0, 0], alpha=0.05)
    plt.scatter(Xt[num_pts:, 0], Xt[num_pts:, 1], c=[1, 0, 0])
    plt.savefig("debug.png")
    import ipdb;ipdb.set_trace()

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
#             plot_pts(key_h.detach().cpu().numpy(), key_h2.detach().cpu().numpy())
#             key_h2, indices = key_h2.squeeze(0), indices.squeeze(0)
#             val_h2 = val_h[indices]
#             attn_val_h = (F.softmax(torch.matmul(query_h, key_h2.transpose(-2, -1)) / math.sqrt(d_k), dim=-1) * val_h2.permute(1, 0)).sum(-1)
#             attn_val_b.append(attn_val_h)
#         attn_val_b = torch.stack(attn_val_b, dim=0)
#         attn_val.append(attn_val_b)
#     attn_val = torch.stack(attn_val, dim=0).unsqueeze(-1)
#     return attn_val


from fast_pytorch_kmeans import KMeans
def attention_group_keys(query, key, value, mask=None, dropout=None, bank_size=2048):

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    batch_size, num_heads, num_v = query.size(0), query.size(1), value.size(2)
    num_k = key.size(2)
    attn_val = []
    for b in range(batch_size):
        attn_val_b = []
        for head_idx in range(num_heads):
            kmeans = KMeans(n_clusters=bank_size, mode='cosine', init_method="kmeans++")
            key_h = key[b, head_idx, ...]
            query_h, val_h = query[b, head_idx, ...], value[b, head_idx, ...]
            pos_mask = val_h.squeeze(-1) > 0
            val_h_pos, val_h_neg = val_h[pos_mask], val_h[torch.logical_not(pos_mask)]
            key_h_pos, key_h_neg = key_h[pos_mask], key_h[torch.logical_not(pos_mask)]
            if key_h_pos.size(0) > bank_size:
                assignments_pos = kmeans.fit_predict(key_h_pos.detach()).cuda()
                centroids_pos = kmeans.centroids.cuda()
            else:
                assignments_pos = torch.arange(0, len(key_h_pos)).long().cuda()
                centroids_pos = key_h_pos.clone()

            # torch.cuda.set_device(current_device)
            # nan_indices = np.where(np.isnan(centroids).any(-1))[0]
            # mask = torch.from_numpy(np.logical_not(np.in1d(assignments, nan_indices))).bool().cuda()
            # assignments = torch.from_numpy(assignments.astype(np.int32)).long().cuda()
            # assignments, key_h, val_h = assignments[mask], key_h[mask], val_h[mask]

            cluster_ids_x_pos = assignments_pos
            cluster_centers = centroids_pos# [np.logical_not(np.isnan(centroids).any(-1))]
            # cluster_centers = torch.from_numpy(cluster_centers).float().cuda()
            val_h2_pos = torch.zeros(centroids_pos.size(0), 1).float().cuda()
            for ind in range(centroids_pos.size(0)):
                val_h2_pos[ind] = torch.nan_to_num(torch.mean(val_h_pos[cluster_ids_x_pos == ind], dim=0), nan=0.)
            key_h2_pos = cluster_centers.cuda()
            assignments_neg = kmeans.fit_predict(key_h_neg.detach()).cuda()
            centroids_neg = kmeans.centroids.cuda()
            cluster_ids_x_neg = assignments_neg
            cluster_centers = centroids_neg
            val_h2_neg = torch.zeros(bank_size, 1).float().cuda()
            for ind in range(bank_size):
                val_h2_neg[ind] = torch.nan_to_num(torch.mean(val_h_neg[cluster_ids_x_neg == ind], dim=0), nan=0.)
            key_h2_neg = cluster_centers.cuda()
            key_h2 = torch.cat((key_h2_pos, key_h2_neg), dim=0)
            val_h2 = torch.cat((val_h2_pos, val_h2_neg), dim=0)
            # plot_pts(key_h.detach().cpu().numpy(), key_h2.detach().cpu().numpy())
            attn_val_h = (F.softmax(torch.matmul(query_h, key_h2.transpose(-2, -1)) / math.sqrt(d_k), dim=-1) * val_h2.permute(1, 0)).sum(-1)
            attn_val_b.append(attn_val_h)
        attn_val_b = torch.stack(attn_val_b, dim=0)
        attn_val.append(attn_val_b)
    attn_val = torch.stack(attn_val, dim=0).unsqueeze(-1)
    return attn_val


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     # rand_idx = torch.randint(low=0, high=key.size(2), size=(2000,))
#     # key, value = key[:, :, rand_idx, :], value[:, :, rand_idx, :]
#     keys = key.view(key.size(0), key.size(1), -1, query.size(2), key.size(-1)).cpu()
#     values = value.view(value.size(0), value.size(1), -1, query.size(2), value.size(-1)).cpu()
#     for i in range(keys.size(2)):
#         key = keys[:, :, i, :, :]
#         value = values[:, :, i, :, :]
#         scores = torch.matmul(query.cpu(), key.cpu().transpose(-2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         p_attn = F.softmax(scores, dim=-1)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         value = torch.matmul(p_attn, value.cpu()).cuda()
#         break
#     return value, p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # rand_idx = torch.randint(low=0, high=key.size(2), size=(2000,))
    # key, value = key[:, :, rand_idx, :], value[:, :, rand_idx, :]

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    value = torch.matmul(p_attn, value).cuda()
    return value, p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
