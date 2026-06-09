"""
test_variants.py — scRareGCR 模块消融与组合测试
不修改 model.py / main.py 等任何已有文件.
用法:
  python test_variants.py --sweep --epochs 50
"""
import argparse, json, os, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, silhouette_score,
)
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import scanpy as sc
import anndata as ad
from model import Model, ZINBLoss
from config import config
from utils import setup_seed, device

warnings.filterwarnings('ignore')

VARIANTS = {
    'V0':    {'gamma': 1.0, 'rbeta': 0.0, 'cluster': 'kl',  'contrast': 'graph',   'readout': 'kmeans',  'desc': '原版scAGCR基线'},
    'V1':    {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'kl',  'contrast': 'density', 'readout': 'kmeans',  'desc': '当前scRareGCR'},
    'V-O':   {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'ot',  'contrast': 'density', 'readout': 'kmeans',  'desc': 'OT聚类+density-CL'},
    'V-P':   {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'kl',  'contrast': 'proto',   'readout': 'kmeans',  'desc': 'KL+Prototype对比'},
    'V-L':   {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'kl',  'contrast': 'density', 'readout': 'leiden',  'desc': 'KL+density-CL+Leiden'},
    'V-OP':  {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'ot',  'contrast': 'proto',   'readout': 'kmeans',  'desc': 'OT+Prototype'},
    'V-OL':  {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'ot',  'contrast': 'density', 'readout': 'leiden',  'desc': 'OT+density-CL+Leiden'},
    'V-OPL': {'gamma': 1.5, 'rbeta': 1.0, 'cluster': 'ot',  'contrast': 'proto',   'readout': 'leiden',  'desc': 'OT+Prototype+Leiden'},
}
CHULI = "/home/liyang/BioJiaheWang/RARECELL/data/chuli"
DEFAULT_DATASETS = ["Pollen", "GSE45719", "GSE67835", "GSE75688", "GSE81861"]

LABEL_KEYS = ["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]

class CellDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.y = np.asarray(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_data(path):
    adata = sc.read_h5ad(path)
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
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float32)
    uniq = sorted(np.unique(y_str))
    lm = {v: i for i, v in enumerate(uniq)}
    y_int = np.array([lm[v] for v in y_str])
    is_rare = None
    if 'is_rare' in adata.obs.columns:
        is_rare = adata.obs['is_rare'].values.astype(int)
    return X, y_str, y_int, is_rare, len(uniq)

def sinkhorn_loss(z, centers, epsilon=0.05, iters=3, tau=0.1):
    cost = torch.cdist(z, centers)
    Q = torch.exp(-cost / epsilon)
    Q = Q / (Q.sum() + 1e-12)
    for _ in range(iters):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
    log_p = F.log_softmax(-cost / tau, dim=1)
    return -torch.mean(torch.sum(Q.detach() * log_p, dim=1))

class PrototypeContrastive(torch.nn.Module):
    def __init__(self, n_proto, dim, tau=0.3):
        super().__init__()
        self.prototypes = torch.nn.Parameter(torch.randn(n_proto, dim))
        torch.nn.init.xavier_uniform_(self.prototypes)
        self.tau = tau
    def forward(self, z):
        z_n = F.normalize(z, dim=1)
        p_n = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(z_n, p_n.T) / self.tau
        with torch.no_grad():
            Q = torch.exp(sim / 0.05)
            Q = Q / (Q.sum() + 1e-12)
            for _ in range(3):
                Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
                Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        log_p = F.log_softmax(sim, dim=1)
        return -torch.mean(torch.sum(Q * log_p, dim=1))

def adaptive_leiden(z, res_list=None):
    if res_list is None:
        res_list = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    adata = ad.AnnData(X=z.copy())
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
    best_score, best_labels = -2, None
    for res in res_list:
        sc.tl.leiden(adata, resolution=res, key_added='tmp')
        labels = adata.obs['tmp'].astype(int).values
        n_cl = len(set(labels))
        if n_cl < 2 or n_cl > len(z) // 3: continue
        try:
            n_s = min(2000, len(z))
            idx = np.random.choice(len(z), n_s, replace=False) if len(z) > n_s else np.arange(len(z))
            s = silhouette_score(z[idx], labels[idx])
            if s > best_score: best_score = s; best_labels = labels.copy()
        except: pass
    if best_labels is None:
        sc.tl.leiden(adata, resolution=0.5, key_added='tmp')
        best_labels = adata.obs['tmp'].astype(int).values
    return best_labels

def cluster_acc(yt, yp):
    yt, yp = np.asarray(yt, np.int64), np.asarray(yp, np.int64)
    D = max(yp.max(), yt.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(yp.size): w[yp[i], yt[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(r, c)) / yp.size

def rare_f1(y_str, pred_labels, is_rare_true, threshold=0.05):
    N = len(pred_labels)
    counts = Counter(pred_labels.tolist() if hasattr(pred_labels, 'tolist') else list(pred_labels))
    rare_cl = {k for k, v in counts.items() if v < N * threshold}
    is_rare_pred = np.array([1 if pred_labels[i] in rare_cl else 0 for i in range(N)])
    if is_rare_true is None:
        tc = Counter(y_str)
        rt = {t for t, c in tc.items() if c < N * threshold}
        if not rt: return {'F1': -1, 'n_pred_rare': int(is_rare_pred.sum())}
        is_rare_true = np.array([1 if y_str[i] in rt else 0 for i in range(N)])
    if is_rare_true.sum() == 0:
        return {'F1': -1, 'n_pred_rare': int(is_rare_pred.sum())}
    return {
        'F1': round(f1_score(is_rare_true, is_rare_pred, zero_division=0), 4),
        'Rec': round(recall_score(is_rare_true, is_rare_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(is_rare_true, is_rare_pred), 4),
        'n_true_rare': int(is_rare_true.sum()),
        'n_pred_rare': int(is_rare_pred.sum()),
    }

def collect_z(model, loader):
    model.eval(); zs, ys = [], []
    with torch.no_grad():
        for bx, by in loader:
            z, *_ = model(bx.float().to(device))
            zs.append(z.cpu().numpy()); ys.append(by)
    return np.vstack(zs), np.hstack(ys)

def init_centers(model, loader, k, seed):
    z, _ = collect_z(model, loader)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(z)
    model.cluster_centers.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
    return km.cluster_centers_

def run_variant(X, y_str, y_int, is_rare_true, n_k, vcfg, seed=1, epochs=50, pretrain=20):
    setup_seed(seed)
    N = len(y_int); input_dim = X.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(X, y_int, test_size=0.2, random_state=1)
    tr_ld = DataLoader(CellDS(Xtr, ytr), batch_size=128, shuffle=True)
    te_ld = DataLoader(CellDS(Xte, yte), batch_size=128, shuffle=False)
    fu_ld = DataLoader(CellDS(X, y_int), batch_size=128, shuffle=False)
    model = Model(
        input_dim, config['graph_head'], config['phi'], config['gcn_dim'], config['mlp_dim'],
        config['prob_feature'], config['prob_edge'], config['tau'], config['alpha'], config['beta'],
        config['dropout'], n_k, config['cluster_alpha'], True,
        gamma_debias=vcfg['gamma'], rare_beta=vcfg['rbeta'],
    ).to(device)
    zinb_fn = ZINBLoss().to(device)
    mae_fn = torch.nn.L1Loss(reduction='mean')
    proto_mod = None
    if vcfg['contrast'] == 'proto':
        proto_mod = PrototypeContrastive(n_k, config['gcn_dim'], tau=0.3).to(device)
    params = list(model.parameters())
    if proto_mod: params += list(proto_mod.parameters())
    opt = torch.optim.Adam(params, lr=config['lr'])
    lam_cl = config['lambda_cl']; lam_clu = config['lambda_cluster']; lam_z = config['lambda_zinb']
    best_ari, best_ep, ckpt = -1, 0, {}
    for ep in range(epochs):
        if ep == pretrain:
            centers = init_centers(model, tr_ld, n_k, seed)
            if proto_mod:
                proto_mod.prototypes.data.copy_(torch.tensor(centers, dtype=torch.float32, device=device))
        model.train()
        if proto_mod: proto_mod.train()
        for bx, _ in tr_ld:
            bx = bx.float().to(device)
            z, x_imp, loss_cl_m, loss_clu_m, mean, disp, pi, q = model(bx)
            mask = torch.where(bx != 0, torch.ones_like(bx), torch.zeros_like(bx))
            loss_mae = mae_fn(mask * x_imp, mask * bx)
            counts = torch.expm1(torch.clamp(bx, min=0.0, max=20.0))
            sf = torch.clamp(counts.sum(1), min=1.0)
            loss_zinb = zinb_fn(counts, mean, disp, pi, sf)
            if vcfg['cluster'] == 'ot' and ep >= pretrain:
                loss_clu = sinkhorn_loss(z, model.cluster_centers)
            else:
                loss_clu = loss_clu_m
            if vcfg['contrast'] == 'proto' and proto_mod is not None:
                loss_con = proto_mod(z)
            else:
                loss_con = loss_cl_m
            cw = 0.0 if ep < pretrain else lam_clu
            loss = loss_mae + lam_cl * loss_con + cw * loss_clu + lam_z * loss_zinb
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            model.eval()
            z_te, y_te = collect_z(model, te_ld)
            if vcfg['readout'] == 'leiden':
                pred_te = adaptive_leiden(z_te)
            else:
                pred_te = KMeans(n_clusters=n_k, random_state=seed, n_init=20).fit_predict(z_te)
            ari = adjusted_rand_score(y_te, pred_te)
            if ari > best_ari:
                best_ari = ari; best_ep = ep
                ckpt = {k: v.clone() for k, v in model.state_dict().items()}
    if ckpt: model.load_state_dict(ckpt)
    z_full, y_full = collect_z(model, fu_ld)
    if vcfg['readout'] == 'leiden':
        pred_full = adaptive_leiden(z_full)
    else:
        pred_full = KMeans(n_clusters=n_k, random_state=seed, n_init=20).fit_predict(z_full)
    ari_f = adjusted_rand_score(y_full, pred_full)
    nmi_f = normalized_mutual_info_score(y_full, pred_full)
    rf = rare_f1(y_str, pred_full, is_rare_true)
    return {'ARI': round(ari_f, 4), 'NMI': round(nmi_f, 4),
            'F1': rf.get('F1', -1), 'Rec': rf.get('Rec', -1), 'MCC': rf.get('MCC', -1),
            'n_pred_rare': rf.get('n_pred_rare', 0), 'best_ep': best_ep}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--variant', type=str, default=None, choices=list(VARIANTS.keys()))
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--pretrain', type=int, default=20)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save_dir', type=str, default='../results/ablation')
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.data_path and args.variant:
        ds = os.path.basename(args.data_path).replace('.h5ad', '')
        vcfg = VARIANTS[args.variant]
        print(f"  {ds} x {args.variant} ({vcfg['desc']})")
        X, ys, yi, ir, nk = load_data(args.data_path)
        res = run_variant(X, ys, yi, ir, nk, vcfg, seed=args.seed, epochs=args.epochs, pretrain=args.pretrain)
        print(f"  ARI={res['ARI']:.4f} NMI={res['NMI']:.4f} F1={res['F1']:.4f} Rec={res['Rec']:.4f} MCC={res['MCC']:.4f}")
        return
    if args.sweep:
        results = []; vtests = list(VARIANTS.keys())
        total = len(args.datasets) * len(vtests); done = 0; t0 = time.time()
        for ds_name in args.datasets:
            h5 = os.path.join(CHULI, f"{ds_name}.h5ad")
            if not os.path.exists(h5): print(f"  跳过 {ds_name}"); continue
            print(f"\n{'='*60}\n数据集: {ds_name}")
            X, ys, yi, ir, nk = load_data(h5)
            print(f"  cells={len(ys)}, k={nk}")
            for vn in vtests:
                done += 1; vcfg = VARIANTS[vn]; t1 = time.time()
                try:
                    res = run_variant(X, ys, yi, ir, nk, vcfg, seed=args.seed, epochs=args.epochs, pretrain=args.pretrain)
                    el = time.time() - t1
                    res.update({'dataset': ds_name, 'variant': vn, 'time': round(el, 1)})
                    results.append(res)
                    print(f"  [{done}/{total}] {vn:>6}: ARI={res['ARI']:.4f} F1={res['F1']:.4f} MCC={res['MCC']:.4f} ({el:.1f}s)")
                except Exception as e:
                    print(f"  [{done}/{total}] {vn:>6}: FAILED - {e}")
        print(f"\n{'='*80}\n消融总表 (epochs={args.epochs}, seed={args.seed})\n{'='*80}")
        hdr = f"{'dataset':<16}"; [hdr := hdr + f"  {v:>7}" for v in vtests]
        print(f"\n--- F1 ---\n{hdr}")
        for ds in args.datasets:
            row = f"{ds:<16}"
            for v in vtests:
                r = next((x for x in results if x['dataset']==ds and x['variant']==v), None)
                row += f"  {r['F1']:>7.4f}" if r and r['F1'] >= 0 else f"  {'--':>7}"
            print(row)
        print(f"\n--- ARI ---\n{hdr}")
        for ds in args.datasets:
            row = f"{ds:<16}"
            for v in vtests:
                r = next((x for x in results if x['dataset']==ds and x['variant']==v), None)
                row += f"  {r['ARI']:>7.4f}" if r else f"  {'--':>7}"
            print(row)
        print(f"\n--- 平均 ---\n{'variant':<8} {'avg_F1':>8} {'avg_ARI':>8} {'avg_MCC':>8}")
        for v in vtests:
            vr = [x for x in results if x['variant'] == v and x['F1'] >= 0]
            if vr:
                print(f"{v:<8} {np.mean([x['F1'] for x in vr]):>8.4f} {np.mean([x['ARI'] for x in vr]):>8.4f} {np.mean([x['MCC'] for x in vr]):>8.4f}")
        out = os.path.join(args.save_dir, f"ablation_seed{args.seed}.json")
        with open(out, 'w') as f: json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果: {out}\n总耗时: {time.time()-t0:.0f}s")
        return
    print("用法: --sweep 扫描全部, 或 --data_path X --variant V 测单个")

if __name__ == '__main__':
    main()
