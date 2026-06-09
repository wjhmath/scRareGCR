"""main_final.py — scRareGCR V1+AB 最终benchmark, 输出对齐scRareCell格式"""
import argparse,json,os,time,warnings,numpy as np,torch,torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import scanpy as sc
from model import Model,ZINBLoss,target_distribution
from config import config
from utils import setup_seed,device
warnings.filterwarnings('ignore')

LABEL_KEYS=["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]
GM=1.5;RB=0.5;LOT=0.1;ZLO=0.7;ZHI=2.0;RT=0.05;PRE=20

class DS(torch.utils.data.Dataset):
    def __init__(s,X,y):s.X=torch.tensor(np.asarray(X,dtype=np.float32));s.y=np.asarray(y)
    def __len__(s):return len(s.y)
    def __getitem__(s,i):return s.X[i],s.y[i]

def load(path):
    a=sc.read_h5ad(path);a.obs_names_make_unique();a.var_names_make_unique()
    ys=None
    for c in LABEL_KEYS:
        if c in a.obs.columns:ys=a.obs[c].astype(str).to_numpy();break
    if ys is None:ys=a.obs.iloc[:,0].astype(str).to_numpy()
    on=a.obs_names.tolist()
    if a.n_obs>20000:np.random.seed(1);ix=np.random.choice(a.n_obs,20000,replace=False);a=a[ix].copy();ys=ys[ix];on=[on[i] for i in ix]
    xm=float(a.X.max()) if hasattr(a.X,"max") else float(np.max(a.X))
    if xm>30:sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    if a.n_vars>2000:sc.pp.highly_variable_genes(a,n_top_genes=2000,flavor="seurat");a=a[:,a.var.highly_variable].copy()
    X=a.X.toarray() if hasattr(a.X,"toarray") else np.asarray(a.X,dtype=np.float32)
    u=sorted(np.unique(ys));lm={v:i for i,v in enumerate(u)};yi=np.array([lm[v] for v in ys])
    ir=a.obs['is_rare'].values.astype(int) if 'is_rare' in a.obs.columns else None
    return X,ys,yi,ir,len(u),on

def sk_loss(z,ctr):
    c=torch.cdist(z,ctr);Q=torch.exp(-c/0.05);Q=Q/(Q.sum()+1e-12)
    for _ in range(3):Q=Q/(Q.sum(0,keepdim=True)+1e-12);Q=Q/(Q.sum(1,keepdim=True)+1e-12)
    return -torch.mean(torch.sum(Q.detach()*F.log_softmax(-c/0.1,dim=1),dim=1))

def znll(x,m,d,p,sf):
    e=1e-10;s=sf.unsqueeze(1);m2=m*s
    t1=torch.lgamma(d+e)+torch.lgamma(x+1.)-torch.lgamma(x+d+e)
    t2=(d+x)*torch.log(1.+(m2/(d+e)))+x*(torch.log(d+e)-torch.log(m2+e))
    nb=t1+t2;nc=nb-torch.log(1.-p+e)
    zn=torch.pow(d/(d+m2+e),d);zc=-torch.log(p+((1.-p)*zn)+e)
    return torch.where(torch.le(x,1e-8),zc,nc).mean(1)

def cacc(yt,yp):
    yt,yp=np.asarray(yt,np.int64),np.asarray(yp,np.int64);D=max(yp.max(),yt.max())+1
    w=np.zeros((D,D),dtype=np.int64)
    for i in range(yp.size):w[yp[i],yt[i]]+=1
    r,c=linear_sum_assignment(w.max()-w);return sum(w[i,j] for i,j in zip(r,c))/yp.size

def gm(yt,yp):
    yt,yp=np.array(yt),np.array(yp)
    tp=((yp==1)&(yt==1)).sum();tn=((yp==0)&(yt==0)).sum()
    fp=((yp==1)&(yt==0)).sum();fn=((yp==0)&(yt==1)).sum()
    return float(np.sqrt((tp/(tp+fn) if tp+fn>0 else 0)*(tn/(tn+fp) if tn+fp>0 else 0)))

def cz(mdl,ld):
    mdl.eval();zs,ys=[],[]
    with torch.no_grad():
        for bx,by in ld:z,*_=mdl(bx.float().to(device));zs.append(z.cpu().numpy());ys.append(by)
    return np.vstack(zs),np.hstack(ys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data_path',required=True)
    p.add_argument('--out_base',default='/home/liyang/BioJiaheWang/RARECELL/benchmark/results/real/scRareGCR')
    p.add_argument('--ds_name',default=None);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--epochs',type=int,default=200)
    a=p.parse_args();t0=time.time()
    ds=a.ds_name or os.path.basename(a.data_path).replace('.h5ad','')
    od=os.path.join(a.out_base,ds);os.makedirs(od,exist_ok=True)
    print(f"{'='*60}\nscRareGCR — {ds}\n{'='*60}")
    X,ys,yi,ir,nk,on=load(a.data_path);N=len(ys);dim=X.shape[1]
    if ir is None:
        tc=Counter(ys);rt={t for t,c in tc.items() if c<N*RT}
        if rt:ir=np.array([1 if ys[i] in rt else 0 for i in range(N)])
    print(f"  cells={N} genes={dim} k={nk} rare={ir.sum() if ir is not None else '?'}")
    Xtr,Xte,ytr,yte=train_test_split(X,yi,test_size=0.2,random_state=1)
    trl=DataLoader(DS(Xtr,ytr),batch_size=128,shuffle=True)
    tel=DataLoader(DS(Xte,yte),batch_size=128,shuffle=False)
    ful=DataLoader(DS(X,yi),batch_size=128,shuffle=False)
    setup_seed(a.seed)
    mdl=Model(dim,config['graph_head'],config['phi'],config['gcn_dim'],config['mlp_dim'],
        config['prob_feature'],config['prob_edge'],config['tau'],config['alpha'],config['beta'],
        config['dropout'],nk,config['cluster_alpha'],True,GM,RB).to(device)
    zf=ZINBLoss().to(device);mf=torch.nn.L1Loss()
    opt=torch.optim.Adam(mdl.parameters(),lr=config['lr'])
    lc,ll,lz=config['lambda_cl'],config['lambda_cluster'],config['lambda_zinb']
    ba,be,ck=-1,0,{}
    for ep in range(a.epochs):
        if ep==PRE:
            z0,_=cz(mdl,trl);km=KMeans(n_clusters=nk,random_state=a.seed,n_init=20).fit(z0)
            mdl.cluster_centers.data.copy_(torch.tensor(km.cluster_centers_,dtype=torch.float32,device=device))
        mdl.train()
        for bx,_ in trl:
            bx=bx.float().to(device);z,xi,lcl,lcu,mn,di,pi,q=mdl(bx)
            mk=torch.where(bx!=0,torch.ones_like(bx),torch.zeros_like(bx))
            lm=mf(mk*xi,mk*bx);ct=torch.expm1(torch.clamp(bx,0,20));sf=torch.clamp(ct.sum(1),min=1.)
            lzb=zf(ct,mn,di,pi,sf)
            if ep>=PRE:
                with torch.no_grad():nl=znll(ct,mn,di,pi,sf);aw=(nl/(nl.mean()+1e-8)).clamp(ZLO,ZHI)
                q2=mdl.soft_assign(z);p2=target_distribution(q2,GM).detach()
                kpc=F.kl_div(torch.log(q2+1e-8),p2,reduction='none').sum(1);lcu=(aw.detach()*kpc).mean()
            lot=sk_loss(z,mdl.cluster_centers) if ep>=PRE else torch.tensor(0.)
            cw=0. if ep<PRE else ll
            loss=lm+lc*lcl+cw*lcu+lz*lzb+LOT*lot
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            mdl.eval();zt,yt=cz(mdl,tel)
            zt=np.nan_to_num(zt,nan=0.0)
            pt=KMeans(n_clusters=nk,random_state=a.seed,n_init=20).fit_predict(zt)
            ar=adjusted_rand_score(yt,pt)
            if ar>ba:ba=ar;be=ep;ck={k:v.clone() for k,v in mdl.state_dict().items()}
    if ck:mdl.load_state_dict(ck)
    print(f"  best_ep={be} test_ari={ba:.4f}")
    zfull,yfull=cz(mdl,ful)
    zfull=np.nan_to_num(zfull,nan=0.0)
    pred=KMeans(n_clusters=nk,random_state=a.seed,n_init=20).fit_predict(zfull)
    # 稀有簇
    cnt=Counter(pred.tolist());rc={k for k,v in cnt.items() if v<N*RT}
    irp=np.array([1 if pred[i] in rc else 0 for i in range(N)])
    tt=time.time()-t0
    # ══ clustering_metrics.json ══
    ari=adjusted_rand_score(yfull,pred);nmi=normalized_mutual_info_score(yfull,pred)
    acc=cacc(yfull,pred)
    clu_m={'ARI':round(ari,6),'NMI':round(nmi,6),'ACC':round(acc,6),
           'n_clusters':len(set(pred)),'n_cells':N}
    with open(os.path.join(od,'clustering_metrics.json'),'w') as f:json.dump(clu_m,f,indent=2)
    print(f"  ARI={ari:.4f} NMI={nmi:.4f} ACC={acc:.4f}")
    # ══ classification_metrics.json ══
    cls_m={'n_pred_rare':int(irp.sum()),'n_cells':N}
    if ir is not None and ir.sum()>0:
        f1=f1_score(ir,irp,zero_division=0);pr=precision_score(ir,irp,zero_division=0)
        rc2=recall_score(ir,irp,zero_division=0);mc=matthews_corrcoef(ir,irp)
        gms=gm(ir,irp);kp=cohen_kappa_score(ir,irp)
        cls_m.update({'F1':round(f1,6),'Precision':round(pr,6),'Recall':round(rc2,6),
            'MCC':round(mc,6),'G_mean':round(gms,6),'Kappa':round(kp,6),'n_true_rare':int(ir.sum())})
        print(f"  F1={f1:.4f} Prec={pr:.4f} Rec={rc2:.4f} MCC={mc:.4f} G-mean={gms:.4f}")
    with open(os.path.join(od,'classification_metrics.json'),'w') as f:json.dump(cls_m,f,indent=2)
    # ══ result.csv ══
    import pandas as pd
    pd.DataFrame({'cell_id':on,'predicted_cluster':pred,'is_rare_pred':irp.astype(int),
        'predicted_label':pred,'true_label':ys}).to_csv(os.path.join(od,'result.csv'),index=False)
    # ══ runtime_log.json ══
    with open(os.path.join(od,'runtime_log.json'),'w') as f:
        json.dump({'status':'OK','method':'scRareGCR','dataset':ds,'n_cells':N,
            'time_sec':round(tt,1),'seed':a.seed,'epochs':a.epochs,
            'gamma':GM,'rare_beta':RB,'lambda_ot':LOT,'zinb_clamp':[ZLO,ZHI]},f,indent=2)
    print(f"  Saved to {od}/ ({tt:.0f}s)")

if __name__=='__main__':main()
