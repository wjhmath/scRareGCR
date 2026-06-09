import torch
import math
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.nn.conv import TransformerConv
from utils import device


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2
        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        if ridge_lambda > 0:
            result += ridge_lambda * torch.square(pi)
        return torch.mean(result)


class MeanAct(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, dropout=0):
        super(GraphConstructor, self).__init__()
        assert input_dim % h == 0
        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))
        return adj, attns

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def DataAug(x, adj, prob_feature, prob_edge):
    batch_size = x.shape[0]
    input_dim = x.shape[1]
    tensor_p = torch.ones((batch_size, input_dim)) * (1 - prob_feature)
    mask_feature = torch.bernoulli(tensor_p).to(device)
    tensor_p = torch.ones((batch_size, batch_size)) * (1 - prob_edge)
    mask_edge = torch.bernoulli(tensor_p).to(device)
    return mask_feature * x, mask_edge * adj


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


# ===== 替换点3: density-aware contrastive =====
def cl_loss(z, z_aug, adj, tau, hidden_norm=True, rare_weight=None):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))
    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)
    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count
    if rare_weight is not None:
        return -torch.sum(rare_weight * loss) / (rare_weight.sum() + 1e-12)
    return -torch.mean(loss, 0)


def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True, rare_weight=None):
    return (alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm, rare_weight)
            + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm, rare_weight))


# ===== 替换点1: frequency-debiased target distribution =====
def target_distribution(q, gamma=1.0):
    f = torch.sum(q, dim=0, keepdim=True)
    weight = q ** 2 / (f ** gamma + 1e-12)
    return (weight.t() / torch.sum(weight, dim=1, keepdim=True).t()).t()




# ===== OT 均衡正则 (Sinkhorn optimal transport) =====
def sinkhorn_loss(z, centers, epsilon=0.05, sinkhorn_iters=3, tau=0.1):
    """最优运输聚类正则: 质量守恒约束防止稀有簇被清空"""
    cost = torch.cdist(z, centers)
    Q = torch.exp(-cost / epsilon)
    Q = Q / (Q.sum() + 1e-12)
    for _ in range(sinkhorn_iters):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
    log_p = F.log_softmax(-cost / tau, dim=1)
    return -torch.mean(torch.sum(Q.detach() * log_p, dim=1))


# ===== ZINB 异常度计算 (per-cell NLL) =====
def zinb_nll_per_cell(x, mean, disp, pi, scale_factor):
    """每个细胞的 ZINB 负对数似然, 用于异常反馈权重"""
    eps = 1e-10
    sf = scale_factor.unsqueeze(1)
    mean = mean * sf
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2
    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
    return result.mean(dim=1)  # per-cell mean across genes

class Model(nn.Module):
    def __init__(self, input_dim, graph_head, phi, gcn_dim, mlp_dim,
                 prob_feature, prob_edge, tau, alpha, beta, dropout,
                 n_clusters, cluster_alpha=1.0, use_graph=True,
                 gamma_debias=1.5, rare_beta=1.0):
        super(Model, self).__init__()
        self.use_graph = use_graph
        self.prob_feature = prob_feature
        self.prob_edge = prob_edge
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.cluster_alpha = cluster_alpha
        self.gamma_debias = gamma_debias
        self.rare_beta = rare_beta
        self.graphconstructor = GraphConstructor(input_dim, graph_head, phi, dropout=0)
        self.transformer = TransformerConv(input_dim, gcn_dim, heads=4, concat=False, dropout=dropout)
        self.w_imp = nn.Linear(gcn_dim, input_dim)
        self.mlp = nn.Linear(gcn_dim, mlp_dim)
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, gcn_dim))
        self.decoder_hidden = nn.Sequential(nn.Linear(gcn_dim, gcn_dim), nn.ReLU())
        self.dec_mean = nn.Sequential(nn.Linear(gcn_dim, input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(gcn_dim, input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(gcn_dim, input_dim), nn.Sigmoid())
        nn.init.xavier_uniform_(self.cluster_centers)
        self.alpha_refine = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(p=dropout)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.cluster_alpha)
        q = q ** ((self.cluster_alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def forward(self, x):
        x = self.dropout(x)
        if self.use_graph:
            adj_init, attn_init = self.graphconstructor(x, x)
            adj = adj_init - torch.diag_embed(adj_init.diag())
        else:
            B = x.shape[0]
            adj = torch.zeros(B, B, device=device)
            attn_init = adj
        edge_index = torch.nonzero(adj == 1).T
        x_aug, adj_aug = DataAug(x, adj, self.prob_feature, self.prob_edge)
        edge_index_aug = torch.nonzero(adj_aug == 1).T
        z, (edge_index_out, trans_attn) = self.transformer(x, edge_index, return_attention_weights=True)
        z_aug, _ = self.transformer(x_aug, edge_index_aug, return_attention_weights=True)
        if self.use_graph and trans_attn is not None and edge_index_out.shape[1] > 0:
            B = x.shape[0]
            trans_attn_mat = torch.zeros(B, B, device=device)
            trans_attn_mat[edge_index_out[0], edge_index_out[1]] = trans_attn.mean(dim=-1) if trans_attn.dim() > 1 else trans_attn.squeeze()
            refined_attn = torch.sigmoid(self.alpha_refine) * trans_attn_mat + (1 - torch.sigmoid(self.alpha_refine)) * attn_init
            refined_attn = torch.sigmoid(refined_attn)
            adj_refined = torch.where(refined_attn >= self.graphconstructor.phi, torch.ones_like(refined_attn), torch.zeros_like(refined_attn))
            adj = adj_refined - torch.diag_embed(adj_refined.diag())

        # ===== 替换点3: density-aware rare weight =====
        with torch.no_grad():
            deg = adj.sum(1)
            w = 1.0 / (deg + 1.0)
            w = w / (w.mean() + 1e-12)
            w = w ** self.rare_beta
            rare_weight = w.detach()

        x_imp = self.w_imp(z)
        z_mlp = self.mlp(z)
        z_mlp_aug = self.mlp(z_aug)
        loss_cl = final_cl_loss(self.alpha, self.beta, z_mlp, z_mlp_aug, adj, adj_aug,
                                self.tau, hidden_norm=True, rare_weight=rare_weight)
        q = self.soft_assign(z)
        p = target_distribution(q, self.gamma_debias).detach()
        loss_cluster = F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
        h = self.decoder_hidden(z)
        mean = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h)
        return z, x_imp, loss_cl, loss_cluster, mean, disp, pi, q
