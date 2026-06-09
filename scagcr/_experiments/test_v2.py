"""test_v2.py — V1基础上叠加新模块测试: OT正则/ZINB反馈/稀有门控"""
import argparse,json,os,time,warnings,numpy as np,torch,torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score,recall_score,matthews_corrcoef,silhouette_score
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import scanpy as sc,anndata as ad
from model import Model,ZINBLoss,target_distribution
from config import config
from utils import setup_seed,device
warnings.filterwarnings('ignore')

CHULI="/home/liyang/BioJiaheWang/RARECELL/data/chuli"
LABEL_KEYS=["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]
VARIANTS={
    'V1':     {'ot':0.0,'zinb_fb':False,'gate':0.0,'desc':'V1基线'},
    'V1+A1':  {'ot':0.1,'zinb_fb':False,'gate':0.0,'desc':'V1+OT(0.1)'},
    'V1+A3':  {'ot':0.3,'zinb_fb':False,'gate':0.0,'desc':'V1+OT(0.3)'},
    'V1+A5':  {'ot':0.5,'zinb_fb':False,'gate':0.0,'desc':'V1+OT(0.5)'},
    'V1+B':   {'ot':0.0,'zinb_fb':True, 'gate':0.0,'desc':'V1+ZINB反馈'},
    'V1+C1':  {'ot':0.0,'zinb_fb':False,'gate':1.0,'desc':'V1+门控(1.0)'},
    'V1+C2':  {'ot':0.0,'zinb_fb':False,'gate':2.0,'desc':'V1+门控(2.0)'},
    'V1+AB':  {'ot':0.3,'zinb_fb':True, 'gate':0.0,'desc':'V1+OT+ZINB'},
    'V1+AC':  {'ot':0.3,'zinb_fb':False,'gate':1.0,'desc':'V1+OT+门控'},
    'V1+BC':  {'ot':0.0,'zinb_fb':True, 'gate':1.0,'desc':'V1+ZINB+门控'},
    'V1+ABC': {'ot':0.3,'zinb_fb':True, 'gate':1.0,'desc':'全家桶'},
}

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
    eps=1e-10;sf=sf.unsqueeze(1);mean=mean*sf
    t1=torch.lgamma(disp+eps)+torch.lgamma(x+1.)-torch.lgamma(x+disp+eps)
    t2=(disp+x)*torch.log(1.+(mean/(disp+eps)))+x*(torch.log(disp+eps)-torch.log(mean+eps))
    nb=t1+t2;nb_c=nb-torch.log(1.-pi+eps)
    z_nb=torch.pow(disp/(disp+mean+eps),disp);z_c=-torch.log(pi+((1.-pi)*z_nb)+eps)
    return torch.where(torch.le(x,1e-8),z_c,nb_c).mean(dim=1)

class RareGate(torch.nn.Module):
    def __init__(s,dim,alpha=1.0):
        super().__init__();s.alpha=alpha
        s.net=torch.nn.Sequential(torch.nn.Linear(dim,64),torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Sigmoid())
    def forward(s,z):
        g=s.net(z)
        return z*(1.0+s.alpha*(1.0-g)),-torch.var(g)*0.1

def cluster_acc(yt,yp):
    yt,yp=np.asarray(yt,np.int64),np.asarray(yp,np.int64);D=max(yp.max(),yt.max())+1
    w=np.zeros((D,D),dtype=np.int64)
    for i in range(yp.size):w[yp[i],yt[i]]+=1
    r,c=linear_sum_assignment(w.max()-w);return sum(w[i,j] for i,j in zip(r,c))/yp.size

def rare_f1(ys,pl,irt,th=0.05):
    N=len(pl);ct=Counter(list(pl));rc={k for k,v in ct.items() if v<N*th}
    irp=np.array([1 if pl[i] in rc else 0 for i in range(N)])
    if irt is None:
        tc=Counter(ys);rt={t for t,c in tc.items() if c<N*th}
        if not rt:return{'F1':-1}
        irt=np.array([1 if ys[i] in rt else 0 for i in range(N)])
    if irt.sum()==0:return{'F1':-1}
    return{'F1':round(f1_score(irt,irp,zero_division=0),4),'Rec':round(recall_score(irt,irp,zero_division=0),4),
           'MCC':round(matthews_corrcoef(irt,irp),4)}

def collect_z(model,loader,gate=None):
    model.eval();zs,ys=[],[]
    with torch.no_grad():
        for bx,by in loader:
            z,*_=model(bx.float().to(device))
            if gate:z,_=gate(z)
            zs.append(z.cpu().numpy());ys.append(by)
    return np.vstack(zs),np.hstack(ys)

def init_centers(model,loader,k,seed):
    z,_=collect_z(model,loader)
    km=KMeans(n_clusters=k,random_state=seed,n_init=20).fit(z)
    model.cluster_centers.data.copy_(torch.tensor(km.cluster_centers_,dtype=torch.float32,device=device))

def run(X,ys,yi,ir,nk,cfg,seed=1,epochs=50,pretrain=20):
    setup_seed(seed);dim=X.shape[1]
    Xtr,Xte,ytr,yte=train_test_split(X,yi,test_size=0.2,random_state=1)
    trl=DataLoader(CellDS(Xtr,ytr),batch_size=128,shuffle=True)
    tel=DataLoader(CellDS(Xte,yte),batch_size=128,shuffle=False)
    ful=DataLoader(CellDS(X,yi),batch_size=128,shuffle=False)
    model=Model(dim,config['graph_head'],config['phi'],config['gcn_dim'],config['mlp_dim'],
        config['prob_feature'],config['prob_edge'],config['tau'],config['alpha'],config['beta'],
        config['dropout'],nk,config['cluster_alpha'],True,gamma_debias=1.5,rare_beta=1.0).to(device)
    zinb_fn=ZINBLoss().to(device);mae_fn=torch.nn.L1Loss()
    gate_mod=RareGate(config['gcn_dim'],cfg['gate']).to(device) if cfg['gate']>0 else None
    params=list(model.parameters())
    if gate_mod:params+=list(gate_mod.parameters())
    opt=torch.optim.Adam(params,lr=config['lr'])
    lc,lcl,lz=config['lambda_cl'],config['lambda_cluster'],config['lambda_zinb']
    best_ari,best_ep,ckpt=-1,0,{}
    for ep in range(epochs):
        if ep==pretrain:init_centers(model,trl,nk,seed)
        model.train()
        if gate_mod:gate_mod.train()
        for bx,_ in trl:
            bx=bx.float().to(device)
            z,x_imp,loss_cl,loss_clu,mean,disp,pi,q=model(bx)
            mask=torch.where(bx!=0,torch.ones_like(bx),torch.zeros_like(bx))
            l_mae=mae_fn(mask*x_imp,mask*bx)
            counts=torch.expm1(torch.clamp(bx,min=0.,max=20.));sf=torch.clamp(counts.sum(1),min=1.)
            l_zinb=zinb_fn(counts,mean,disp,pi,sf)
            z_cl=z;gate_reg=0
            if gate_mod:z_cl,gate_reg=gate_mod(z)
            aw=None
            if cfg['zinb_fb'] and ep>=pretrain:
                with torch.no_grad():
                    nll=zinb_nll_per_cell(counts,mean,disp,pi,sf)
                    aw=(nll/(nll.mean()+1e-8)).clamp(0.5,3.0)
            if cfg['gate']>0 or cfg['zinb_fb']:
                q2=model.soft_assign(z_cl);p2=target_distribution(q2,1.5).detach()
                if aw is not None:
                    kl_pc=F.kl_div(torch.log(q2+1e-8),p2,reduction='none').sum(1)
                    loss_clu=(aw*kl_pc).mean()
                else:
                    loss_clu=F.kl_div(torch.log(q2+1e-8),p2,reduction='batchmean')
            l_ot=sinkhorn_loss(z,model.cluster_centers) if cfg['ot']>0 and ep>=pretrain else torch.tensor(0.)
            cw=0. if ep<pretrain else lcl
            loss=l_mae+lc*loss_cl+cw*loss_clu+lz*l_zinb+cfg['ot']*l_ot+gate_reg
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            model.eval()
            zt,yt=collect_z(model,tel,gate_mod)
            pt=KMeans(n_clusters=nk,random_state=seed,n_init=20).fit_predict(zt)
            ari=adjusted_rand_score(yt,pt)
            if ari>best_ari:best_ari=ari;best_ep=ep;ckpt={k:v.clone() for k,v in model.state_dict().items()}
    if ckpt:model.load_state_dict(ckpt)
    zf,yf=collect_z(model,ful,gate_mod)
    pf=KMeans(n_clusters=nk,random_state=seed,n_init=20).fit_predict(zf)
    rf=rare_f1(ys,pf,ir)
    return{'ARI':round(adjusted_rand_score(yf,pf),4),'NMI':round(normalized_mutual_info_score(yf,pf),4),
           'F1':rf.get('F1',-1),'Rec':rf.get('Rec',-1),'MCC':rf.get('MCC',-1),'ep':best_ep}

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--save_dir',type=str,default='../results/ablation_v2')
    a=p.parse_args();os.makedirs(a.save_dir,exist_ok=True)
    results=[];vtests=list(VARIANTS.keys());total=len(a.datasets)*len(vtests);done=0
    for ds in a.datasets:
        h5=os.path.join(CHULI,f"{ds}.h5ad")
        if not os.path.exists(h5):print(f"跳过 {ds}");continue
        print(f"\n{'='*60}\n数据集: {ds}");X,ys,yi,ir,nk=load_data(h5);print(f"  cells={len(ys)}, k={nk}")
        for vn in vtests:
            done+=1;cfg=VARIANTS[vn];t1=time.time()
            try:
                r=run(X,ys,yi,ir,nk,cfg,seed=a.seed,epochs=a.epochs)
                el=time.time()-t1;r.update({'dataset':ds,'variant':vn,'time':round(el,1)});results.append(r)
                print(f"  [{done}/{total}] {vn:>7}: ARI={r['ARI']:.4f} F1={r['F1']:.4f} MCC={r['MCC']:.4f} ({el:.0f}s)")
            except Exception as e:print(f"  [{done}/{total}] {vn:>7}: FAIL - {e}")
    print(f"\n{'='*80}\n--- F1 ---")
    hdr=f"{'ds':<16}";[hdr:=hdr+f" {v:>7}" for v in vtests];print(hdr)
    for ds in a.datasets:
        row=f"{ds:<16}"
        for v in vtests:
            r=next((x for x in results if x['dataset']==ds and x['variant']==v),None)
            row+=f" {r['F1']:>7.4f}" if r and r['F1']>=0 else f" {'--':>7}"
        print(row)
    print(f"\n--- ARI ---");print(hdr)
    for ds in a.datasets:
        row=f"{ds:<16}"
        for v in vtests:
            r=next((x for x in results if x['dataset']==ds and x['variant']==v),None)
            row+=f" {r['ARI']:>7.4f}" if r else f" {'--':>7}"
        print(row)
    print(f"\n--- 平均 ---\n{'var':<8}{'F1':>8}{'ARI':>8}{'MCC':>8}")
    for v in vtests:
        vr=[x for x in results if x['variant']==v and x['F1']>=0]
        if vr:print(f"{v:<8}{np.mean([x['F1'] for x in vr]):>8.4f}{np.mean([x['ARI'] for x in vr]):>8.4f}{np.mean([x['MCC'] for x in vr]):>8.4f}")
    out=os.path.join(a.save_dir,f"v2_seed{a.seed}.json")
    with open(out,'w') as f:json.dump(results,f,indent=2);print(f"\n保存: {out}")

if __name__=='__main__':main()
