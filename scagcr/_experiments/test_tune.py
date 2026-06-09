"""test_tune.py — V1+AB 参数调优: OT权重 × ZINB反馈策略 × gamma"""
import argparse,json,os,time,warnings,numpy as np,torch,torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,f1_score,recall_score,matthews_corrcoef
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import scanpy as sc
from model import Model,ZINBLoss,target_distribution
from config import config
from utils import setup_seed,device
warnings.filterwarnings('ignore')

CHULI="/home/liyang/BioJiaheWang/RARECELL/data/chuli"
LABEL_KEYS=["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]

class CellDS(torch.utils.data.Dataset):
    def __init__(s,X,y):s.X=torch.tensor(np.asarray(X,dtype=np.float32));s.y=np.asarray(y)
    def __len__(s):return len(s.y)
    def __getitem__(s,i):return s.X[i],s.y[i]

def load_data(path):
    a=sc.read_h5ad(path);a.obs_names_make_unique();a.var_names_make_unique()
    ys=None
    for c in LABEL_KEYS:
        if c in a.obs.columns:ys=a.obs[c].astype(str).to_numpy();break
    if ys is None:ys=a.obs.iloc[:,0].astype(str).to_numpy()
    if a.n_obs>20000:np.random.seed(1);idx=np.random.choice(a.n_obs,20000,replace=False);a=a[idx].copy();ys=ys[idx]
    xm=float(a.X.max()) if hasattr(a.X,"max") else float(np.max(a.X))
    if xm>30:sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    if a.n_vars>2000:sc.pp.highly_variable_genes(a,n_top_genes=2000,flavor="seurat");a=a[:,a.var.highly_variable].copy()
    X=a.X.toarray() if hasattr(a.X,"toarray") else np.asarray(a.X,dtype=np.float32)
    u=sorted(np.unique(ys));lm={v:i for i,v in enumerate(u)};yi=np.array([lm[v] for v in ys])
    ir=a.obs['is_rare'].values.astype(int) if 'is_rare' in a.obs.columns else None
    return X,ys,yi,ir,len(u)

def sinkhorn_loss(z,centers,eps=0.05,iters=3,tau=0.1):
    c=torch.cdist(z,centers);Q=torch.exp(-c/eps);Q=Q/(Q.sum()+1e-12)
    for _ in range(iters):Q=Q/(Q.sum(0,keepdim=True)+1e-12);Q=Q/(Q.sum(1,keepdim=True)+1e-12)
    return -torch.mean(torch.sum(Q.detach()*F.log_softmax(-c/tau,dim=1),dim=1))

def zinb_nll_per_cell(x,mean,disp,pi,sf):
    eps=1e-10;sf2=sf.unsqueeze(1);m=mean*sf2
    t1=torch.lgamma(disp+eps)+torch.lgamma(x+1.)-torch.lgamma(x+disp+eps)
    t2=(disp+x)*torch.log(1.+(m/(disp+eps)))+x*(torch.log(disp+eps)-torch.log(m+eps))
    nb=t1+t2;nb_c=nb-torch.log(1.-pi+eps)
    z_nb=torch.pow(disp/(disp+m+eps),disp);z_c=-torch.log(pi+((1.-pi)*z_nb)+eps)
    return torch.where(torch.le(x,1e-8),z_c,nb_c).mean(dim=1)

def compute_zinb_weight(nll, strategy):
    """不同的 ZINB 异常权重策略"""
    if strategy == 'off':
        return None
    elif strategy.startswith('clamp_'):
        # 固定 clamp: clamp_1.5 means clamp(0.8, 1.5)
        hi = float(strategy.split('_')[1])
        lo = max(0.5, 1.0 - (hi - 1.0) * 0.5)  # 对称缩放
        w = (nll / (nll.mean() + 1e-8)).clamp(lo, hi)
        return w
    elif strategy == 'rank':
        # 自适应排名法: 无需手调 clamp
        rank = nll.argsort().argsort().float()  # 0 到 N-1
        rank = rank / (rank.max() + 1e-8)       # 归一到 [0, 1]
        w = 0.5 + rank                           # [0.5, 1.5], 稀有=高
        return w
    elif strategy == 'softrank':
        # 软排名: sigmoid 平滑
        z_score = (nll - nll.mean()) / (nll.std() + 1e-8)
        w = 0.5 + torch.sigmoid(z_score)         # [0.5, 1.5]
        return w
    return None

def rare_f1(ys,pl,irt,th=0.05):
    N=len(pl);ct=Counter(list(pl));rc={k for k,v in ct.items() if v<N*th}
    irp=np.array([1 if pl[i] in rc else 0 for i in range(N)])
    if irt is None:
        tc=Counter(ys);rt={t for t,c in tc.items() if c<N*th}
        if not rt:return -1,-1,-1
        irt=np.array([1 if ys[i] in rt else 0 for i in range(N)])
    if irt.sum()==0:return -1,-1,-1
    return(round(f1_score(irt,irp,zero_division=0),4),
           round(recall_score(irt,irp,zero_division=0),4),
           round(matthews_corrcoef(irt,irp),4))

def collect_z(model,loader):
    model.eval();zs,ys=[],[]
    with torch.no_grad():
        for bx,by in loader:
            z,*_=model(bx.float().to(device));zs.append(z.cpu().numpy());ys.append(by)
    return np.vstack(zs),np.hstack(ys)

def init_centers(model,loader,k,seed):
    z,_=collect_z(model,loader)
    km=KMeans(n_clusters=k,random_state=seed,n_init=20).fit(z)
    model.cluster_centers.data.copy_(torch.tensor(km.cluster_centers_,dtype=torch.float32,device=device))

def run(X,ys,yi,ir,nk,params,seed=1,epochs=100,pretrain=20):
    setup_seed(seed);dim=X.shape[1]
    Xtr,Xte,ytr,yte=train_test_split(X,yi,test_size=0.2,random_state=1)
    trl=DataLoader(CellDS(Xtr,ytr),batch_size=128,shuffle=True)
    tel=DataLoader(CellDS(Xte,yte),batch_size=128,shuffle=False)
    ful=DataLoader(CellDS(X,yi),batch_size=128,shuffle=False)
    model=Model(dim,config['graph_head'],config['phi'],config['gcn_dim'],config['mlp_dim'],
        config['prob_feature'],config['prob_edge'],config['tau'],config['alpha'],config['beta'],
        config['dropout'],nk,config['cluster_alpha'],True,
        gamma_debias=params['gamma'],rare_beta=params['rbeta']).to(device)
    zinb_fn=ZINBLoss().to(device);mae_fn=torch.nn.L1Loss()
    opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
    lc,lcl,lz=config['lambda_cl'],config['lambda_cluster'],config['lambda_zinb']
    best_ari,best_ep,ckpt=-1,0,{}
    for ep in range(epochs):
        if ep==pretrain:init_centers(model,trl,nk,seed)
        model.train()
        for bx,_ in trl:
            bx=bx.float().to(device)
            z,x_imp,loss_cl,loss_clu,mean,disp,pi,q=model(bx)
            mask=torch.where(bx!=0,torch.ones_like(bx),torch.zeros_like(bx))
            l_mae=mae_fn(mask*x_imp,mask*bx)
            counts=torch.expm1(torch.clamp(bx,min=0.,max=20.));sf=torch.clamp(counts.sum(1),min=1.)
            l_zinb=zinb_fn(counts,mean,disp,pi,sf)
            # ZINB 异常反馈
            if params['zinb_strategy']!='off' and ep>=pretrain:
                with torch.no_grad():
                    nll=zinb_nll_per_cell(counts,mean,disp,pi,sf)
                    aw=compute_zinb_weight(nll,params['zinb_strategy'])
                if aw is not None:
                    q2=model.soft_assign(z);p2=target_distribution(q2,params['gamma']).detach()
                    kl_pc=F.kl_div(torch.log(q2+1e-8),p2,reduction='none').sum(1)
                    loss_clu=(aw.detach()*kl_pc).mean()
            # OT 正则
            l_ot=torch.tensor(0.,device=device)
            if params['lambda_ot']>0 and ep>=pretrain:
                l_ot=sinkhorn_loss(z,model.cluster_centers)
            cw=0. if ep<pretrain else lcl
            loss=l_mae+lc*loss_cl+cw*loss_clu+lz*l_zinb+params['lambda_ot']*l_ot
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            model.eval();zt,yt=collect_z(model,tel)
            pt=KMeans(n_clusters=nk,random_state=seed,n_init=20).fit_predict(zt)
            ari=adjusted_rand_score(yt,pt)
            if ari>best_ari:best_ari=ari;best_ep=ep;ckpt={k:v.clone() for k,v in model.state_dict().items()}
    if ckpt:model.load_state_dict(ckpt)
    zf,yf=collect_z(model,ful)
    pf=KMeans(n_clusters=nk,random_state=seed,n_init=20).fit_predict(zf)
    ari_f=adjusted_rand_score(yf,pf)
    f1,rec,mcc=rare_f1(ys,pf,ir)
    return{'ARI':round(ari_f,4),'F1':f1,'Rec':rec,'MCC':mcc,'ep':best_ep}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=1)
    a=p.parse_args()

    # 参数网格
    grid=[]
    for gamma in [1.5, 2.0, 2.5]:
      for rbeta in [0.5, 1.0]:
        for lam_ot in [0.0, 0.1, 0.3]:
          for zs in ['off','clamp_1.3','clamp_1.8','clamp_2.5','rank','softrank']:
            name=f"g{gamma}_rb{rbeta}_ot{lam_ot}_{zs}"
            grid.append({'name':name,'gamma':gamma,'rbeta':rbeta,
                         'lambda_ot':lam_ot,'zinb_strategy':zs})

    total=len(a.datasets)*len(grid);done=0;t0=time.time()
    all_results=[]
    print(f"参数组合: {len(grid)}, 数据集: {len(a.datasets)}, 总实验: {total}")

    for ds in a.datasets:
        h5=os.path.join(CHULI,f"{ds}.h5ad")
        if not os.path.exists(h5):print(f"跳过 {ds}");continue
        X,ys,yi,ir,nk=load_data(h5)
        print(f"\n{'='*70}\n{ds} (cells={len(ys)}, k={nk})")
        ds_results=[]
        for cfg in grid:
            done+=1;t1=time.time()
            try:
                r=run(X,ys,yi,ir,nk,cfg,seed=a.seed,epochs=a.epochs)
                el=time.time()-t1
                r.update({'dataset':ds,'params':cfg['name']})
                ds_results.append(r);all_results.append(r)
                if done%10==0:
                    print(f"  [{done}/{total}] {cfg['name']}: F1={r['F1']:.4f} ARI={r['ARI']:.4f} ({el:.0f}s)")
            except Exception as e:
                print(f"  [{done}/{total}] {cfg['name']}: FAIL {e}")

        # 该数据集 top5
        valid=[x for x in ds_results if x['F1']>=0]
        if valid:
            by_f1=sorted(valid,key=lambda x:-x['F1'])[:5]
            print(f"\n  Top5 by F1:")
            for i,r in enumerate(by_f1):
                print(f"    #{i+1} {r['params']}: F1={r['F1']:.4f} ARI={r['ARI']:.4f} MCC={r['MCC']:.4f}")
            by_comp=sorted(valid,key=lambda x:-(x['F1']*0.4+x['MCC']*0.3+x['ARI']*0.3))[:5]
            print(f"  Top5 综合(F1×0.4+MCC×0.3+ARI×0.3):")
            for i,r in enumerate(by_comp):
                score=r['F1']*0.4+r['MCC']*0.3+r['ARI']*0.3
                print(f"    #{i+1} {r['params']}: score={score:.4f} F1={r['F1']:.4f} ARI={r['ARI']:.4f}")

    # 跨数据集汇总: 哪组参数平均最好
    print(f"\n{'='*70}\n跨数据集 Top10 参数组合\n{'='*70}")
    from collections import defaultdict
    param_scores=defaultdict(list)
    for r in all_results:
        if r['F1']>=0:
            param_scores[r['params']].append(r['F1']*0.4+r['MCC']*0.3+r['ARI']*0.3)
    avg_scores={k:np.mean(v) for k,v in param_scores.items() if len(v)>=len(a.datasets)*0.5}
    for rank,(k,v) in enumerate(sorted(avg_scores.items(),key=lambda x:-x[1])[:10],1):
        print(f"  #{rank:>2} {k:<40} avg_score={v:.4f} (n={len(param_scores[k])})")

    print(f"\n总耗时: {time.time()-t0:.0f}s")
    # 保存
    out=f"../results/tune_seed{a.seed}.json"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,'w') as f:json.dump(all_results,f,indent=2)
    print(f"保存: {out}")

if __name__=='__main__':main()
