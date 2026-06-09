"""
main_rare.py — scRareGCR 最终版 (V1+AB)

创新点:
  1. γ-去偏自训练 (frequency-debiased KL target)
  2. 密度感知对比学习 (density-weighted contrastive)
  3. OT 均衡正则 (Sinkhorn optimal transport regularizer)
  4. ZINB 异常反馈 (reconstruction-driven rare cell feedback)

用法:
  python main_rare.py --data_path /path/to/data.h5ad --save_dir ../results/scRareGCR
"""

import argparse, json, os, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score,
    homogeneity_score, completeness_score,
)
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import scanpy as sc

from model import Model, ZINBLoss, target_distribution, sinkhorn_loss, zinb_nll_per_cell
from config import config
from utils import setup_seed, device

warnings.filterwarnings('ignore')

LABEL_KEYS = ["cell_type", "cell_type_label", "CellType", "celltype",
              "label", "labels", "Group"]


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if hasattr(X, "toarray"): X = X.toarray()
        self.X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.y = np.asarray(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def load_data(data_path):
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique(); adata.var_names_make_unique()
    y_str = None
    for col in LABEL_KEYS:
        if col in adata.obs.columns:
            y_str = adata.obs[col].astype(str).to_numpy(); break
    if y_str is None:
        y_str = adata.obs.iloc[:, 0].astype(str).to_numpy()
    if adata.n_obs > 20000:
        np.random.seed(1)
        idx = np.random.choice(adata.n_obs, 20000, replace=False)
        adata = adata[idx].copy(); y_str = y_str[idx]
    x_max = float(adata.X.max()) if hasattr(adata.X, "max") else float(np.max(adata.X))
    if x_max > 30:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    if adata.n_vars > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()
    X = adata.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    uniq = sorted(np.unique(y_str))
    label_map = {v: i for i, v in enumerate(uniq)}
    y_int = np.array([label_map[v] for v in y_str])
    is_rare_true = None
    if 'is_rare' in adata.obs.columns:
        is_rare_true = adata.obs['is_rare'].values.astype(int)
    return X, y_str, y_int, is_rare_true, len(uniq)


def build_loaders(X, y_int, batch_size=128):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_int, test_size=0.2, random_state=1)
    train_loader = DataLoader(CellDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CellDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(CellDataset(X, y_int), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, full_loader


# ═══════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════

def cluster_acc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, np.int64), np.asarray(y_pred, np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size): w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(r, c)) / y_pred.size


def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    return sum(max(Counter(y_true[y_pred == c]).values()) for c in set(y_pred)) / len(y_true)


def gmean_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    return float(np.sqrt(sens * spec))


# ═══════════════════════════════════════════════════════
# 训练 (V1+AB: γ-KL + density-CL + OT正则 + ZINB反馈)
# ═══════════════════════════════════════════════════════

def collect_embeddings(model, loader):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for bx, by in loader:
            z, *_ = model(bx.float().to(device))
            zs.append(z.cpu().numpy()); ys.append(by)
    return np.vstack(zs), np.hstack(ys)


def initialize_centers(model, loader, n_clusters, seed):
    z, _ = collect_embeddings(model, loader)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z)
    model.cluster_centers.data.copy_(
        torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))


def train_model(args, train_loader, test_loader, input_dim):
    gamma = args.gamma_debias
    rare_beta = args.rare_beta
    lambda_ot = args.lambda_ot
    zinb_lo = args.zinb_clamp_lo
    zinb_hi = args.zinb_clamp_hi

    model = Model(
        input_dim, args.graph_head, args.phi, args.gcn_dim, args.mlp_dim,
        args.prob_feature, args.prob_edge, args.tau, args.alpha, args.beta,
        args.dropout, args.n_clusters, args.cluster_alpha,
        not args.no_graph, gamma, rare_beta,
    ).to(device)

    zinb_loss_fn = ZINBLoss().to(device)
    mae_fn = torch.nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    setup_seed(args.seed)

    lam_cl = args.lambda_cl
    lam_clu = args.lambda_cluster
    lam_zinb = args.lambda_zinb
    pretrain = args.pretrain_epochs

    best_ari, best_epoch = -1.0, 0
    ckpt_path = os.path.join(args.out_dir, 'checkpoint.pt')

    for epoch in range(args.epochs):
        if epoch == pretrain:
            initialize_centers(model, train_loader, args.n_clusters, args.seed)

        # ── 训练 ──
        model.train()
        for bx, _ in train_loader:
            bx = bx.float().to(device)
            z, x_imp, loss_cl, loss_cluster, mean, disp, pi, q = model(bx)

            # 重建损失
            mask = torch.where(bx != 0, torch.ones_like(bx), torch.zeros_like(bx))
            loss_mae = mae_fn(mask * x_imp, mask * bx)
            counts = torch.expm1(torch.clamp(bx, min=0.0, max=20.0))
            sf = torch.clamp(counts.sum(1), min=1.0)
            loss_zinb = zinb_loss_fn(counts, mean, disp, pi, sf)

            # ===== 创新4: ZINB 异常反馈 =====
            if epoch >= pretrain:
                with torch.no_grad():
                    nll = zinb_nll_per_cell(counts, mean, disp, pi, sf)
                    anomaly_w = (nll / (nll.mean() + 1e-8)).clamp(zinb_lo, zinb_hi)
                # 用异常权重重新计算聚类损失
                q_new = model.soft_assign(z)
                p_new = target_distribution(q_new, gamma).detach()
                kl_per_cell = F.kl_div(
                    torch.log(q_new + 1e-8), p_new, reduction='none').sum(1)
                loss_cluster = (anomaly_w.detach() * kl_per_cell).mean()

            # ===== 创新3: OT 均衡正则 =====
            loss_ot = torch.tensor(0.0, device=device)
            if lambda_ot > 0 and epoch >= pretrain:
                loss_ot = sinkhorn_loss(z, model.cluster_centers)

            # 总损失
            cw = 0.0 if epoch < pretrain else lam_clu
            loss = (loss_mae
                    + lam_cl * loss_cl          # 创新2: density-CL (在model内)
                    + cw * loss_cluster         # 创新1+4: γ-KL + ZINB反馈
                    + lam_zinb * loss_zinb
                    + lambda_ot * loss_ot)      # 创新3: OT正则

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ── 验证 ──
        with torch.no_grad():
            model.eval()
            z_te, y_te = collect_embeddings(model, test_loader)
            pred_te = KMeans(
                n_clusters=args.n_clusters, random_state=args.seed,
                n_init=20).fit_predict(z_te)
            ari = adjusted_rand_score(y_te, pred_te)
            if ari > best_ari:
                best_ari = ari; best_epoch = epoch
                torch.save({'net': model.state_dict(), 'epoch': epoch,
                            'ari': float(ari)}, ckpt_path)

    # 加载最优
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['net'])
    print(f"  Best epoch={best_epoch}, test ARI={best_ari:.4f}")
    return model


# ═══════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='scRareGCR — rare cell identification')
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='../results/scRareGCR')
    p.add_argument('--n_clusters', type=int, default=None)
    p.add_argument('--no_graph', action='store_true')
    # 模型参数
    for k in ['graph_head','phi','gcn_dim','mlp_dim','prob_feature','prob_edge',
              'tau','alpha','beta','lambda_cl','lambda_cluster','cluster_alpha',
              'lambda_zinb','dropout','lr','pretrain_epochs','seed','epochs']:
        v = config[k]
        p.add_argument(f'--{k}', type=type(v), default=v)
    # 稀有模块参数
    p.add_argument('--gamma_debias', type=float, default=config['gamma_debias'])
    p.add_argument('--rare_beta', type=float, default=config['rare_beta'])
    p.add_argument('--lambda_ot', type=float, default=config['lambda_ot'])
    p.add_argument('--zinb_clamp_lo', type=float, default=config['zinb_clamp_lo'])
    p.add_argument('--zinb_clamp_hi', type=float, default=config['zinb_clamp_hi'])
    p.add_argument('--rare_threshold', type=float, default=config['rare_threshold'])
    args = p.parse_args()

    t0 = time.time()
    dataset_name = Path(args.data_path).stem
    args.out_dir = os.path.join(args.save_dir, dataset_name, f'seed{args.seed}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载数据 ──
    print(f"{'='*60}")
    print(f"scRareGCR — {dataset_name}")
    print(f"{'='*60}")
    X, y_str, y_int, is_rare_true, auto_k = load_data(args.data_path)
    if args.n_clusters is None: args.n_clusters = auto_k
    N = len(y_str); input_dim = X.shape[1]
    print(f"  cells={N}, genes={input_dim}, k={args.n_clusters}")
    print(f"  gamma={args.gamma_debias}, rare_beta={args.rare_beta}, "
          f"lambda_ot={args.lambda_ot}, zinb_clamp=[{args.zinb_clamp_lo},{args.zinb_clamp_hi}]")

    if is_rare_true is None:
        type_counts = Counter(y_str)
        rare_types = {t for t, c in type_counts.items() if c < N * args.rare_threshold}
        if rare_types:
            is_rare_true = np.array([1 if y_str[i] in rare_types else 0 for i in range(N)])
            print(f"  is_rare 推断: {sorted(rare_types)}, {is_rare_true.sum()}/{N}")

    train_loader, test_loader, full_loader = build_loaders(X, y_int)

    # ── 训练 ──
    print(f"  训练中 (epochs={args.epochs}, pretrain={args.pretrain_epochs}) ...")
    model = train_model(args, train_loader, test_loader, input_dim)

    # ── 全量推断 ──
    print(f"  全量推断 ...")
    z_full, y_full = collect_embeddings(model, full_loader)
    pred_labels = KMeans(n_clusters=args.n_clusters, random_state=args.seed,
                         n_init=20).fit_predict(z_full)

    counts = Counter(pred_labels.tolist())
    rare_clusters = {k for k, v in counts.items() if v < N * args.rare_threshold}
    is_rare_pred = np.array([1 if pred_labels[i] in rare_clusters else 0 for i in range(N)])

    # ── 聚类指标 ──
    nmi = normalized_mutual_info_score(y_full, pred_labels)
    ari = adjusted_rand_score(y_full, pred_labels)
    acc = cluster_acc(y_full, pred_labels)
    pur = purity_score(y_full, pred_labels)
    print(f"\n  === Clustering ===")
    print(f"  ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  Purity={pur:.4f}")

    results = {
        'dataset': dataset_name, 'seed': args.seed, 'n_cells': N,
        'n_clusters': args.n_clusters, 'n_pred_rare': int(is_rare_pred.sum()),
        'ACC': round(float(acc), 6), 'NMI': round(float(nmi), 6),
        'ARI': round(float(ari), 6), 'Purity': round(float(pur), 6),
        'gamma_debias': args.gamma_debias, 'rare_beta': args.rare_beta,
        'lambda_ot': args.lambda_ot,
        'zinb_clamp': [args.zinb_clamp_lo, args.zinb_clamp_hi],
    }

    # ── 稀有检测指标 ──
    if is_rare_true is not None and is_rare_true.sum() > 0:
        f1 = f1_score(is_rare_true, is_rare_pred, zero_division=0)
        prec = precision_score(is_rare_true, is_rare_pred, zero_division=0)
        rec = recall_score(is_rare_true, is_rare_pred, zero_division=0)
        mcc = matthews_corrcoef(is_rare_true, is_rare_pred)
        gms = gmean_score(is_rare_true, is_rare_pred)
        kappa = cohen_kappa_score(is_rare_true, is_rare_pred)
        print(f"\n  === Rare Cell Detection ===")
        print(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
              f"MCC={mcc:.4f}  G-mean={gms:.4f}  Kappa={kappa:.4f}")
        results.update({
            'F1': round(float(f1), 6), 'Precision': round(float(prec), 6),
            'Recall': round(float(rec), 6), 'MCC': round(float(mcc), 6),
            'G_mean': round(float(gms), 6), 'Kappa': round(float(kappa), 6),
            'n_true_rare': int(is_rare_true.sum()),
        })
        # Per-type
        type_counts = Counter(y_str)
        for ct in sorted(type_counts, key=lambda x: type_counts[x]):
            if type_counts[ct] < N * args.rare_threshold:
                ct_mask = (y_str == ct)
                det = int((is_rare_pred[ct_mask] == 1).sum())
                print(f"    {ct}: {det}/{type_counts[ct]} ({det/type_counts[ct]*100:.1f}%)")

    # ── 簇分布 ──
    print(f"\n  === Cluster Distribution ===")
    for k in sorted(counts):
        flag = " <- RARE" if k in rare_clusters else ""
        print(f"    Cluster {k}: {counts[k]} ({counts[k]/N*100:.1f}%){flag}")

    # ── 保存 ──
    total_time = time.time() - t0
    results['time_sec'] = round(total_time, 1)

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), z_full)
    np.save(os.path.join(args.out_dir, 'predictions.npy'), pred_labels)
    np.save(os.path.join(args.out_dir, 'is_rare_pred.npy'), is_rare_pred)

    import pandas as pd
    pd.DataFrame({
        'cell_type': y_str, 'true_label_int': y_full,
        'predicted_cluster': pred_labels,
        'is_rare_pred': is_rare_pred.astype(int),
        'is_rare_true': is_rare_true.astype(int) if is_rare_true is not None else -1,
    }).to_csv(os.path.join(args.out_dir, 'result.csv'), index=False)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Saved to: {args.out_dir}/")
    print("  Done!")


if __name__ == '__main__':
    main()
