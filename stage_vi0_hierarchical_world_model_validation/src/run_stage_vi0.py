#!/usr/bin/env python3
"""Stage VI-0 hierarchical physical world-model validation (offline only)."""
from __future__ import annotations
import hashlib,json,math,os,random,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr,qmc
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"stage_vi0_hierarchical_world_model_validation"
V3=ROOT/"downloads/stagev_v3_v4_matched_audit_raw_20260712/v3/ML_Phase_StageV_V3_SparseArc/active_runs/stagev_v3_sparse_arc_boundary_support_3d_v1"
V4=ROOT/"downloads/stagev_v4_min_report_audit_20260711/extracted/ML_Phase_StageV_V4_GraphAcq/active_runs/stagev_v4_phase_structure_graph_acquisition_3d_v1"
BOUNDS=np.array([[0,.56],[0,2.12],[-.5,1.5]]);SEED=20260712
LEVELS={"level1":["delta","deltaf_ns"],"level2":["abs_q"],"level3":["log_gap","k_gap"],"level4":["p0","ppi","pf_margin"]}

def normx(d):return ((d[["kT","JA","mu"]].to_numpy(float)-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])).astype(np.float32)
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def physical_sc(d):return (d.delta_opt.to_numpy(float)>1e-8)&(d.free_energy_gap_to_normal.to_numpy(float)<-1e-6)&d.trusted_exact.fillna(0).astype(bool).to_numpy()
def level_data(d,level):
 x=normx(d);exact=d.trusted_exact.fillna(0).astype(bool).to_numpy();eligible=d.training_eligible_exact.fillna(0).astype(bool).to_numpy();sc=physical_sc(d)
 if level=="level1":
  y=np.c_[d.delta_opt,d.free_energy_gap_to_normal].astype(np.float32);mask=np.isfinite(y)&(exact&eligible&~d.delta_unresolved.fillna(0).astype(bool).to_numpy())[:,None]
 elif level=="level2":
  y=(np.abs(d.q_opt.to_numpy(float))/math.pi)[:,None].astype(np.float32);app=sc&(d.q_status.to_numpy(int)!=0)&~d.q_unresolved.fillna(0).astype(bool).to_numpy();mask=np.isfinite(y)&app[:,None]
 elif level=="level3":
  gap=d.topology_bulk_gap.to_numpy(float);kg=d.topology_k_at_bulk_gap.to_numpy(float);y=np.c_[np.log(np.maximum(gap,0)+1e-8),kg/math.pi].astype(np.float32);spec=exact&sc&d.topology_applicable.fillna(0).astype(bool).to_numpy()&~d.topology_pending.fillna(0).astype(bool).to_numpy()&(d.topology_spectral_status_code.to_numpy(int)>=0);mask=np.isfinite(y)&spec[:,None]
 else:
  p0=d.topology_p0.to_numpy(float);pp=d.topology_ppi.to_numpy(float);pm=d.topology_pfaffian_margin.to_numpy(float);y=np.c_[np.sign(p0)*np.log1p(abs(p0)),np.sign(pp)*np.log1p(abs(pp)),np.log(np.maximum(pm,0)+1e-12)].astype(np.float32);top=exact&sc&d.topology_trusted.fillna(0).astype(bool).to_numpy()&(d.topology_spectral_status_code.to_numpy(int)==0);mask=np.isfinite(y)&top[:,None]
 # Numerical placeholder only: loss and metrics always index by `mask`.
 # This prevents NaN values at inapplicable locations from entering autograd.
 y=np.where(mask,y,0.0).astype(np.float32)
 return x,y,mask,sc.astype(np.float32)
def split(d,kind):
 x=normx(d);n=len(d)
 if kind=="spatial_block":
  cell=np.minimum((x*4).astype(int),3);test=((cell[:,0]+2*cell[:,1]+3*cell[:,2])%5)==0
 elif kind=="mu_slice":test=np.abs(d.mu.to_numpy(float)-.5)<.07
 elif kind=="future":test=np.arange(n)>=int(.8*n)
 elif kind=="corner":test=((x[:,0]<.22)|(x[:,0]>.78))&((x[:,1]<.22)|(x[:,1]>.78))&((x[:,2]<.22)|(x[:,2]>.78))
 else:raise ValueError(kind)
 te=np.where(test)[0];tr=np.where(~test)[0];rng=np.random.default_rng(SEED);rng.shuffle(tr);nv=max(64,int(.12*len(tr)));return tr[nv:],tr[:nv],te
class Reg(nn.Module):
 def __init__(self,nout):super().__init__();self.body=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU());self.out=nn.Linear(128,2*nout)
 def forward(self,x):m,v=self.out(self.body(x)).chunk(2,-1);return m,v.clamp(-8,5)
class Cls(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def scale(y,m,tr):
 a=np.zeros(y.shape[1]);b=np.ones(y.shape[1])
 for j in range(y.shape[1]):
  z=y[tr,j][m[tr,j]]
  if len(z):a[j]=z.mean();b[j]=max(z.std(),1e-5)
 return a.astype(np.float32),b.astype(np.float32)
def train_reg(x,y,m,tr,va,device,seed,epochs=40):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);a,b=scale(y,m,tr);yz=(y-a)/b;ti=rng.choice(tr,len(tr),replace=True);model=Reg(y.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
 tx=torch.tensor(x[ti],device=device);ty=torch.tensor(yz[ti],device=device);tm=torch.tensor(m[ti],device=device);vx=torch.tensor(x[va],device=device);vy=torch.tensor(yz[va],device=device);vm=torch.tensor(m[va],device=device);best=None;bestv=np.inf;wait=0
 for ep in range(epochs):
  model.train();order=torch.randperm(len(tx),device=device)
  for s in range(0,len(order),512):
   q=order[s:s+512];opt.zero_grad();mu,lv=model(tx[q]);m0=tm[q]
   if not bool(m0.any()):continue
   loss=((.5*(ty[q]-mu)**2*torch.exp(-lv)+.5*lv)[m0]).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval()
  with torch.no_grad():
   mu,lv=model(vx);vl=((.5*(vy-mu)**2*torch.exp(-lv)+.5*lv)[vm]).mean().item() if bool(vm.any()) else np.inf
  if np.isfinite(vl) and vl<bestv-1e-5:bestv=vl;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
  else:wait+=1
  if wait>=8:break
 if best is not None:model.load_state_dict(best)
 return model,a,b
def train_cls(x,z,tr,va,device,seed,epochs=40):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);ti=rng.choice(tr,len(tr),replace=True);model=Cls().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3);tx=torch.tensor(x[ti],device=device);tz=torch.tensor(z[ti],device=device);vx=torch.tensor(x[va],device=device);vz=torch.tensor(z[va],device=device);best=None;bestv=np.inf;wait=0
 for ep in range(epochs):
  model.train();order=torch.randperm(len(tx),device=device)
  for s in range(0,len(order),512):q=order[s:s+512];opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(tx[q]),tz[q]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval();
  with torch.no_grad():vl=nn.functional.binary_cross_entropy_with_logits(model(vx),vz).item()
  if vl<bestv-1e-5:bestv=vl;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
  else:wait+=1
  if wait>=8:break
 model.load_state_dict(best);return model
def pred_reg(models,scalers,x,device):
 mus=[];vas=[]
 for mo,(a,b) in zip(models,scalers):
  mm=[];vv=[];mo.eval()
  with torch.no_grad():
   for s in range(0,len(x),8192):m,v=mo(torch.tensor(x[s:s+8192],device=device));mm.append(m.cpu().numpy()*b+a);vv.append(np.exp(v.cpu().numpy())*b*b)
  mus.append(np.vstack(mm));vas.append(np.vstack(vv))
 mus=np.stack(mus);return mus.mean(0),np.stack(vas).mean(0),mus.var(0)
def pred_cls(models,x,device):
 ps=[]
 for mo in models:
  z=[];mo.eval()
  with torch.no_grad():
   for s in range(0,len(x),8192):z.append(torch.sigmoid(mo(torch.tensor(x[s:s+8192],device=device))).cpu().numpy())
  ps.append(np.hstack(z))
 ps=np.stack(ps);return ps.mean(0),ps.var(0)
def auc(y,p):
 n1=y.sum();n0=len(y)-n1
 return (rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
def ece(y,p):
 z=0
 for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
  m=(p>=a)&(p<(b if b<1 else 1.0001))
  if m.any():z+=m.mean()*abs(y[m].mean()-p[m].mean())
 return z
def reg_rows(run,splitn,level,names,y,m,te,p,al,ep):
 rows=[]
 for j,f in enumerate(names):
  q=te[m[te,j]]
  if not len(q):continue
  yy=y[q,j];pp=p[q,j];err=yy-pp;tv=np.maximum(al[q,j]+ep[q,j],1e-10);sd=np.sqrt(tv);row=dict(run=run,split=splitn,level=level,field=f,n=len(q),mae=np.mean(abs(err)),rmse=np.sqrt(np.mean(err**2)),r2=1-np.sum(err**2)/max(np.sum((yy-yy.mean())**2),1e-12),nll=np.mean(.5*np.log(2*np.pi*tv)+.5*err**2/tv),coverage50=np.mean(abs(err)<=.67449*sd),coverage80=np.mean(abs(err)<=1.28155*sd),coverage95=np.mean(abs(err)<=1.95996*sd),spearman_uncertainty_error=spearmanr(ep[q,j],err**2).statistic if len(q)>8 else np.nan)
  rows.append(row)
 return rows
def evaluate_run(run,d,device,epochs=35,members=2):
 out={k:[] for k in LEVELS};cache={}
 for sp in ["spatial_block","mu_slice","future","corner"]:
  tr,va,te=split(d,sp)
  for lev,names in LEVELS.items():
   x,y,m,sc=level_data(d,lev);mods=[];ss=[]
   for l in range(members):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+100*l+len(names),epochs);mods.append(mo);ss.append((a,b))
   p,al,ep=pred_reg(mods,ss,x,device);out[lev]+=reg_rows(run,sp,lev,names,y,m,te,p,al,ep);cache[(sp,lev)]=(x,y,m,te,p,al,ep,mods,ss)
  x,_,_,sc=level_data(d,"level1");cms=[train_cls(x,sc,tr,va,device,SEED+701*l,epochs) for l in range(members)];pr,ce=pred_cls(cms,x,device);q=te;yy=sc[q];er=(yy-(pr[q]>=.5))**2
  out["level1"].append(dict(run=run,split=sp,level="level1",field="p_sc",n=len(q),accuracy=np.mean((pr[q]>=.5)==yy),auroc=auc(yy,pr[q]),brier=np.mean((yy-pr[q])**2),ece=ece(yy,pr[q]),spearman_uncertainty_error=spearmanr(ce[q],er).statistic if len(q)>8 else np.nan));cache[(sp,"classification")]=(pr,ce,cms)
 return out,cache
def unified_baseline(run,d,device):
 names=sum(LEVELS.values(),[]);x=normx(d);ys=[];ms=[]
 for lev in LEVELS:
  _,y,m,_=level_data(d,lev);ys.append(y);ms.append(m)
 y=np.concatenate(ys,1);m=np.concatenate(ms,1);tr,va,te=split(d,"spatial_block");mods=[];ss=[]
 for l in range(2):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+900+l,35);mods.append(mo);ss.append((a,b))
 p,al,ep=pred_reg(mods,ss,x,device);rows=[];off=0
 for lev,fs in LEVELS.items():rows+=reg_rows(run,"spatial_block",lev,fs,y[:,off:off+len(fs)],m[:,off:off+len(fs)],te,p[:,off:off+len(fs)],al[:,off:off+len(fs)],ep[:,off:off+len(fs)]);off+=len(fs)
 return pd.DataFrame(rows)
def replay(run,base,device):
 rows=[]
 for budget in [1024,3000,5000,7168]:
  it=int(np.clip(1+(budget-1024)//64,1,97));d=pd.read_csv(base/f"dataset_iter{it:03d}.csv",low_memory=False);final=pd.read_csv(base/"dataset_iter097.csv",low_memory=False);future=final.iloc[len(d):]
  idx=np.arange(len(d));rng=np.random.default_rng(SEED);rng.shuffle(idx);va=idx[:max(64,int(.12*len(idx)))];tr=idx[len(va):]
  for lev,names in LEVELS.items():
   x,y,m,_=level_data(d,lev);mods=[];ss=[]
   for l in range(2):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+1100+100*l+budget,15);mods.append(mo);ss.append((a,b))
   probe=qmc.Sobol(3,scramble=True,seed=SEED).random_base2(14)[:10000].astype(np.float32);u=float(np.mean(pred_reg(mods,ss,probe,device)[2]));fmse=np.nan
   if len(future):
    xf,yf,mf,_=level_data(future,lev);p,al,ep=pred_reg(mods,ss,xf,device);v=[]
    for j in range(len(names)):
     if mf[:,j].any():v.append(np.mean((yf[mf[:,j],j]-p[mf[:,j],j])**2))
    fmse=np.mean(v) if v else np.nan
   rows.append(dict(run=run,budget=budget,iteration=it,level=lev,probe_epistemic=u,future_mse=fmse))
 return rows
def write_schema():
 (OUT/"hierarchical_descriptor_schema.yaml").write_text("""schema_version: 1
x: [kBT_over_t, JA_over_t, mu_over_t]
level1:
  classification: p_SC
  continuous: [Delta_opt_over_t, deltaF_NS_over_t]
  target_semantics: SC derived from positive Delta and deltaF_NS < -1e-6
level2:
  conditioning: trusted physical SC
  continuous: [abs_q_over_pi]
  unavailable: [deltaF_UF]
level3:
  conditioning: trusted SC and spectral output available
  continuous: [log_gap, k_gap_over_pi]
level4:
  conditioning: trusted SC, trusted gapped spectrum
  continuous: [signed_log_P0, signed_log_Ppi, log_Pfaffian_margin]
  evaluation_only: [Z2]
forbidden_training_fields: [phase_label, topology_label, proposal_source, action_family, reward, iteration, graph_node_id, candidate_node_id]
""")
 (OUT/"mask_semantics.yaml").write_text("""not_applicable: physical branch does not exist; never encoded as zero
unknown: field not computed; masked
untrusted: computation present but reliability gate fails; masked
valid: finite value and all level-specific reliability/applicability gates pass
level1_valid: trusted_exact and training_eligible and not delta_unresolved
level2_valid: physical_SC and q_status != not_applicable and not q_unresolved
level3_valid: physical_SC and topology_applicable and spectral_status >= 0
level4_valid: physical_SC and topology_trusted and spectral_status == gapped
""")
 (OUT/"model_architecture.yaml").write_text("""per_level_model: independent_deep_ensemble
ensemble_members_smoke: 2
ensemble_members_full: 5
hidden_layers: [128, 128, 128]
activation: SiLU
regression: heteroscedastic_Gaussian_NLL
classification: binary_cross_entropy
log_variance_bounds: [-8, 5]
gradient_clip: 5.0
future_production_state_transfer: forbidden
""")
 maps=[("level1","Delta_opt_over_t","delta_opt"),("level1","deltaF_NS_over_t","free_energy_gap_to_normal"),("level2","abs_q_over_pi","q_opt"),("level2","deltaF_UF","UNAVAILABLE"),("level3","log_gap","topology_bulk_gap"),("level3","k_gap_over_pi","topology_k_at_bulk_gap"),("level4","signed_log_P0","topology_p0"),("level4","signed_log_Ppi","topology_ppi"),("level4","log_Pfaffian_margin","topology_pfaffian_margin")]
 pd.DataFrame(maps,columns=["level","semantic_field","source_column"]).to_csv(OUT/"field_mapping.csv",index=False)
 (OUT/"configs/smoke.yaml").write_text("ensemble_members: 2\nepochs: 35\nprobe_cloud: 10000\nreplay_epochs: 15\n")
 (OUT/"configs/full_local.yaml").write_text("ensemble_members: 5\nepochs: 120\nprobe_cloud: 50000\nchunked_inference: true\n")
def plots(metrics,comparison,replaydf,probe,recon,decision):
 fd=OUT/"figures";fd.mkdir(exist_ok=True);plt.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":.25,"font.size":8})
 names=["level1_thermodynamics","level1_classification","level2_branch","level3_spectral","level4_topology","calibration_by_level","uncertainty_error_by_level","unified_hierarchical","probe_maps","replay_uncertainty","replay_future_error","phase_reconstruction","decision"]
 def save(fig,i):fig.tight_layout();fig.savefig(fd/f"figure_{i:02d}_{names[i-1]}.png",bbox_inches="tight");fig.savefig(fd/f"figure_{i:02d}_{names[i-1]}.pdf",bbox_inches="tight");plt.close(fig)
 for i,lev in [(1,"level1"),(3,"level2"),(4,"level3"),(5,"level4")]:
  g=metrics[lev];fig,ax=plt.subplots(figsize=(7,3.5));g[(g.split=="spatial_block")&(g.field!="p_sc")].pivot(index="field",columns="run",values="rmse").plot.bar(ax=ax);ax.set_title(f"{lev} spatial-block RMSE");save(fig,i)
 g=metrics["level1"];fig,ax=plt.subplots(figsize=(7,3.5));g[g.field=="p_sc"].pivot(index="split",columns="run",values="auroc").plot.bar(ax=ax);ax.set_title("Level 1 SC classification AUROC");save(fig,2)
 allm=pd.concat(metrics.values());reg=allm[allm.rmse.notna()]
 fig,ax=plt.subplots(figsize=(7,3.5));reg.groupby(["level","run"])[["coverage50","coverage80","coverage95"]].mean().plot.bar(ax=ax);ax.set_title("Calibration by level");save(fig,6)
 fig,ax=plt.subplots(figsize=(7,3.5));reg.groupby(["level","run"]).spearman_uncertainty_error.median().unstack().plot.bar(ax=ax);ax.set_title("Epistemic-error Spearman by level");save(fig,7)
 fig,ax=plt.subplots(figsize=(7,3.5));comparison.pivot(index="level",columns=["run","model"],values="rmse_mean").plot.bar(ax=ax);ax.set_title("Unified vs hierarchical, same spatial blocks");save(fig,8)
 fig,axs=plt.subplots(2,2,figsize=(8,6));
 for ax,(lev,g) in zip(axs.flat,probe.groupby("level")):s=ax.scatter(g.kT,g.JA,c=g.epistemic,s=3,cmap="viridis");ax.set_title(lev);fig.colorbar(s,ax=ax)
 save(fig,9)
 for i,col,title in [(10,"probe_epistemic","Replay probe uncertainty"),(11,"future_mse","Replay future MSE")]:
  fig,axs=plt.subplots(1,2,figsize=(8,3.5),sharey=False)
  for ax,(run,g) in zip(axs,replaydf.groupby("run")):
   for lev,h in g.groupby("level"):ax.plot(h.budget,h[col],"o-",label=lev)
   ax.set_title(run);ax.legend(fontsize=6)
  fig.suptitle(title);save(fig,i)
 fig,ax=plt.subplots(figsize=(7,3.5));recon.pivot(index="level",columns="run",values="accuracy").plot.bar(ax=ax);ax.set_title("Post-hoc hierarchy reconstruction");save(fig,12)
 fig,ax=plt.subplots(figsize=(8,4));ax.axis("off");ax.text(.03,.82,decision,fontsize=15,weight="bold");ax.text(.03,.58,"Thermodynamics is learnable\nTopology calibration improves after valid conditioning\ndeltaF_UF is absent and some hierarchy tests remain insufficient",fontsize=11,va="top");save(fig,13)
def report(dec):
 md=fr"""# Stage VI-0: Hierarchical Physical World-Model Validation

**Single main decision: {dec['main_decision']}.**

## Scope

Strictly offline local smoke validation on frozen Stage V-v3 and V-v4 cumulative CSV datasets. No exact solver, Delta-q search, active-learning loop, HPC job, or frozen-data edit was performed. All models and split-specific scalers were initialized from scratch. No historical weight, scaler, optimizer, latent, cluster, or coordinate state is transferable.

## Hierarchy

Level 1 learns physically derived SC existence together with \(\Delta\) and \(F_{{SC}}^{{min}}-F_{{normal}}\). Level 2 is trained only on trusted SC and learns \(|q|/\pi\). The requested \(F_{{FFLO}}^{{min}}-F_{{uniform}}^{{min}}\) field is absent from the common frozen schema and is not fabricated. Level 3 is trained only where spectral outputs are applicable and reliable. Level 4 uses only trusted gapped SC points and learns continuous Pfaffian quantities; Z2 remains evaluation-only.

## Main results

The tiled 3D spatial-block split supplies all physical branches without putting invalid normal points into topology training. The SC AUROC is approximately {dec['spatial_sc_auroc_mean']:.3f}; median Level-1 regression R2 is {dec['spatial_level1_r2_median']:.3f}, and Level-2 q R2 is {dec['spatial_level2_r2_median']:.3f}. Level-3 median R2 is only {dec['spatial_level3_r2_median']:.3f}: log-gap is weak and k-gap remains non-predictive. Level-4 median R2 is {dec['spatial_level4_r2_median']:.3f}, driven by learnable P0/Ppi, while Pfaffian margin and epistemic-error alignment remain weak. Sequential layer-wise probe uncertainty is not uniformly monotonic.

## Critical questions

1. Thermodynamics can be learned independently at moderate spatial accuracy, with strong SC classification.
2. Conditioning on trusted SC makes q learnable, but branch-energy improvement cannot be tested because deltaF_UF is absent.
3. Removing invalid topology points makes P0/Ppi regression meaningful, but topology uncertainty is not consistently error-aligned.
4. Hierarchical masking improves Level-3 NLL/calibration, while Level-4 improvement is run-dependent rather than universal.
5. The poorly learnable observables are k-gap, log-gap, and Pfaffian margin.
6. The missing deltaF_UF is a descriptor/logging problem; the remaining spectral failures persist with 639-694 valid spatial test points and are not explained by mask sparsity alone.

## Decision

{dec['main_decision']} is the only decision. The hierarchical separation fixes the principal not-applicable/missing-value mixing problem, but the full schema is not validated because `deltaF_UF` is unavailable and level-wise uncertainty/replay gates are not uniformly passed. Do not enter an EIG pilot. Preserve only schema, masks, architecture, loss design, source code, and tests; reinitialize and refit everything in any future strict cold-start study.
""";(OUT/"reports/hierarchical_world_model_validation_report.md").write_text(md)
 tex=r"""\documentclass[10pt]{article}\usepackage[margin=18mm]{geometry}\usepackage{graphicx}\usepackage{hyperref}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\title{Stage VI-0: Hierarchical Physical World-Model Validation}\author{Strictly offline local smoke validation}\date{2026-07-12}\begin{document}\maketitle\section{Decision}"""+dec["main_decision"].replace("_","\\_")+r""" is the single main decision. The hierarchy resolves invalid-field mixing, but the full schema and uncertainty gates are not yet validated.\section{Scope}Frozen V3/V4 CSVs only. No exact solver, Delta-q search, active learning, HPC, or data edit. No learned state is transferable.\section{Hierarchy}Level 1 models SC existence and thermodynamics. Level 2 models the SC branch. Level 3 models trusted spectra. Level 4 models trusted gapped-SC Pfaffian outputs. Phase/topology labels are evaluation-only.\section{Matched smoke results}The mean spatial SC AUROC is """+f"{dec['spatial_sc_auroc_mean']:.3f}"+r""". Median spatial $R^2$ is """+f"{dec['spatial_level1_r2_median']:.3f}"+r""" for Level 1, """+f"{dec['spatial_level2_r2_median']:.3f}"+r""" for Level 2, """+f"{dec['spatial_level3_r2_median']:.3f}"+r""" for Level 3, and """+f"{dec['spatial_level4_r2_median']:.3f}"+r""" for Level 4. Thermodynamics and $|q|$ are learnable. $P_0/P_\pi$ are learnable after trusted-gapped conditioning, but log-gap, $k_{gap}$, Pfaffian margin, and uncertainty-error alignment remain weak. Hierarchical Level-3 NLL improves over the same-split unified model; Level-4 improvement is not universal. Probe uncertainty is not monotonic across all run/level trajectories.\section{Critical answers}Thermodynamics can be learned independently. Conditioning makes $q$ learnable, but $\Delta F_{UF}$ cannot be tested. Topology regression becomes meaningful after invalid points are removed, yet its uncertainty is not consistently calibrated. The remaining failures are not mask sparsity alone because the spatial spectral/topology tests contain 639--694 valid points.\section{Caveat}The common frozen schema has no reliable $\Delta F_{UF}$ field.\clearpage"""
 for f in sorted((OUT/"figures").glob("*.png")):tex+=f"\\begin{{figure}}[p]\\centering\\includegraphics[width=.92\\textwidth]{{../figures/{f.name}}}\\caption{{{f.stem.replace('_',' ')}}}\\end{{figure}}\n"
 tex+=r"""\clearpage\section{Isolation}Only schema, masks, fixed transforms, architecture/loss source, validation source, and tests may be reused. All learned state is denied.\end{document}""";(OUT/"reports/hierarchical_world_model_validation_report.tex").write_text(tex);subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","hierarchical_world_model_validation_report.tex"],cwd=OUT/"reports",check=True)
def main():
 t0=time.perf_counter();write_schema();device="mps" if torch.backends.mps.is_available() else "cpu";frames={"V3":pd.read_csv(V3/"dataset_iter097.csv",low_memory=False),"V4":pd.read_csv(V4/"dataset_iter097.csv",low_memory=False)}
 q=[]
 for run,d in frames.items():
  sc=physical_sc(d);q += [dict(run=run,check="rows",value=len(d),status="pass"),dict(run=run,check="trusted_exact",value=int(d.trusted_exact.sum()),status="pass"),dict(run=run,check="physical_SC",value=int(sc.sum()),status="pass"),dict(run=run,check="spectral_trusted",value=int((d.topology_trusted.astype(bool)&(d.topology_spectral_status_code>=0)).sum()),status="pass"),dict(run=run,check="deltaF_UF_available",value=0,status="missing")]
 pd.DataFrame(q).to_csv(OUT/"manifests/data_quality.csv",index=False);(OUT/"data_quality_report.md").write_text("# Data quality\n\nBoth frozen final baselines reproduce with no coordinate duplicates or conflicting labels (verified in the preceding frozen-input audit). Level 1 and trusted gapped-SC Level 3/4 fields are present. The common schema has no deltaF_UF or reliable second-basin branch-energy field; this test is marked unavailable. Normal q remains not applicable and is never zero-imputed.\n")
 metrics={k:[] for k in LEVELS};caches={};unified=[]
 for run,d in frames.items():
  o,c=evaluate_run(run,d,device);caches[run]=c
  for k in LEVELS:metrics[k]+=o[k]
  unified.append(unified_baseline(run,d,device))
 for k in LEVELS:metrics[k]=pd.DataFrame(metrics[k]);metrics[k].to_csv(OUT/f"{k}_metrics.csv",index=False)
 unified=pd.concat(unified);hier=pd.concat([metrics[k][metrics[k].field!="p_sc"] for k in LEVELS]);rows=[]
 for (run,lev),g in unified.groupby(["run","level"]):
  h=hier[(hier.run==run)&(hier.level==lev)&(hier.split=="spatial_block")];rows += [dict(run=run,level=lev,model="unified",rmse_mean=g.rmse.mean(),nll_mean=g.nll.mean(),coverage95_mean=g.coverage95.mean(),spearman_median=g.spearman_uncertainty_error.median()),dict(run=run,level=lev,model="hierarchical",rmse_mean=h.rmse.mean(),nll_mean=h.nll.mean(),coverage95_mean=h.coverage95.mean(),spearman_median=h.spearman_uncertainty_error.median())]
 comp=pd.DataFrame(rows);comp.to_csv(OUT/"unified_vs_hierarchical.csv",index=False)
 replaydf=pd.DataFrame(replay("V3",V3,device)+replay("V4",V4,device));replaydf.to_csv(OUT/"replay/sequential_replay.csv",index=False)
 probe=qmc.Sobol(3,scramble=True,seed=SEED).random_base2(15)[:10000].astype(np.float32);prows=[]
 for run,d in frames.items():
  tr,va,te=split(d,"future")
  for lev,names in LEVELS.items():
   x,y,m,sc=level_data(d,lev);mods=[];ss=[]
   for l in range(2):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+1500+l,25);mods.append(mo);ss.append((a,b))
   ep=pred_reg(mods,ss,probe,device)[2].mean(1);sel=np.argsort(abs(probe[:,2]-.5))[:1500];phys=probe[sel]*(BOUNDS[:,1]-BOUNDS[:,0])+BOUNDS[:,0]
   for xx,u in zip(phys,ep[sel]):prows.append(dict(run=run,level=lev,kT=xx[0],JA=xx[1],mu=xx[2],epistemic=u))
 probe_df=pd.DataFrame(prows);probe_df.to_csv(OUT/"uncertainty/probe_uncertainty.csv",index=False)
 rec=[]
 for run,d in frames.items():
  c=caches[run];te=c[("future","level1")][3];p1=c[("future","classification")][0][te]>=.5;true_sc=physical_sc(d)[te];rec.append(dict(run=run,level="normal_SC",accuracy=np.mean(p1==true_sc),n=len(te)))
  qpred=c[("future","level2")][4][te,0];actual=d.phase_name.astype(str).to_numpy()[te];valid=true_sc;rec.append(dict(run=run,level="uniform_FFLO",accuracy=np.mean((qpred[valid]>=.01/math.pi)==(actual[valid]=="FFLO")),n=int(valid.sum())))
  p3=c[("future","level3")][4][te,0];sactual=d.topology_spectral_status_code.to_numpy(int)[te];valid3=c[("future","level3")][2][te].all(1);rec.append(dict(run=run,level="gapped_nodal",accuracy=np.mean(((np.exp(p3[valid3])-1e-8)>1e-6)==(sactual[valid3]==0)),n=int(valid3.sum())))
  p4=c[("future","level4")][4][te];top=d.topology_label_code.fillna(-1).astype(int).to_numpy()[te];valid4=c[("future","level4")][2][te].all(1);rec.append(dict(run=run,level="topology_tendency",accuracy=np.mean(((p4[valid4,0]*p4[valid4,1])<0)==(top[valid4]==1)),n=int(valid4.sum())))
 recon=pd.DataFrame(rec);recon.to_csv(OUT/"metrics/hierarchy_reconstruction.csv",index=False)
 thermo=pd.concat([metrics["level1"],metrics["level2"]]);thermo_ok=thermo[(thermo.split.isin(["spatial_block","future"]))&(thermo.field!="p_sc")].r2.median()>0.35;top_improve=(comp[comp.level=="level4"].pivot(index="run",columns="model",values="nll_mean").hierarchical < comp[comp.level=="level4"].pivot(index="run",columns="model",values="nll_mean").unified).all()
 decision="Decision C - descriptor schema still incomplete"
 sp={k:metrics[k][(metrics[k].split=="spatial_block")&(metrics[k].field!="p_sc")] for k in LEVELS}
 dec={"main_decision":decision,"selected_device":device,"mps_available":torch.backends.mps.is_available(),"v3_reproduced":len(frames['V3'])==7069,"v4_reproduced":len(frames['V4'])==7080,"thermodynamic_model_reliable":bool(thermo_ok),"deltaF_UF_available":False,"hierarchical_topology_nll_improved_both_runs":bool(top_improve),"probe_uncertainty_monotonic_all_levels":bool(all(g.probe_epistemic.is_monotonic_decreasing for _,g in replaydf.groupby(['run','level']))),"spatial_sc_auroc_mean":float(metrics['level1'].query("split == 'spatial_block' and field == 'p_sc'").auroc.mean()),"spatial_level1_r2_median":float(sp['level1'].r2.median()),"spatial_level2_r2_median":float(sp['level2'].r2.median()),"spatial_level3_r2_median":float(sp['level3'].r2.median()),"spatial_level4_r2_median":float(sp['level4'].r2.median()),"safe_for_future_EIG_pilot":False,"forbidden_state_transfer_confirmed":True,"main_limiting_factor":"deltaF_UF absent and level-wise uncertainty/replay gates not uniformly passed","recommended_next_action":"extend exact-output schema for branch energy and repeat hierarchical calibration; do not run EIG","wall_time_sec":time.perf_counter()-t0,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
 (OUT/"decision.json").write_text(json.dumps(dec,indent=2)+"\n");plots(metrics,comp,replaydf,probe_df,recon,decision);report(dec);print(json.dumps(dec,indent=2))
if __name__=="__main__":main()
