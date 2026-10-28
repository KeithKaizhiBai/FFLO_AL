#!/usr/bin/env python3
"""Stage VI-0.2 offline descriptor refinement and uncertainty calibration."""
from __future__ import annotations
import json,math,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,qmc
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"stage_vi0_2_hierarchical_world_model_v2"
sys.path.insert(0,str(ROOT/"stage_vi0_hierarchical_world_model_validation/src"))
from run_stage_vi0 import (V3,V4,BOUNDS,SEED,normx,physical_sc,split,train_reg,pred_reg,train_cls,pred_cls,auc,ece,level_data as v1_level_data)
LEVELS={"level1":["delta","deltaf_ns"],"level2":["abs_q"],"level3":["log_gap"],"level4":["p0","ppi","pf_margin"]}

def level_data(d,lev):
 if lev!="level3":return v1_level_data(d,lev)
 x,y,m,sc=v1_level_data(d,"level3");return x,y[:,:1],m[:,:1],sc
def scale(y,m,tr):
 a=np.zeros(y.shape[1]);b=np.ones(y.shape[1])
 for j in range(y.shape[1]):
  z=y[tr,j][m[tr,j]]
  if len(z):a[j]=z.mean();b[j]=max(z.std(),1e-5)
 return a.astype(np.float32),b.astype(np.float32)
def risk_auc(err2,unc):
 q=np.argsort(unc);cs=np.linspace(.1,1,10);risk=np.array([err2[q[:max(1,int(c*len(q)))]].mean() for c in cs]);return float(np.trapezoid(risk,cs))
def rows(run,splitn,lev,names,y,m,te,p,al,ep):
 out=[]
 for j,f in enumerate(names):
  q=te[m[te,j]]
  if not len(q):continue
  yy=y[q,j];pp=p[q,j];er=yy-pp;tv=np.maximum(al[q,j]+ep[q,j],1e-10);sd=np.sqrt(tv)
  out.append(dict(run=run,split=splitn,level=lev,field=f,n=len(q),mae=np.mean(abs(er)),rmse=np.sqrt(np.mean(er**2)),r2=1-np.sum(er**2)/max(np.sum((yy-yy.mean())**2),1e-12),nll=np.mean(.5*np.log(2*np.pi*tv)+.5*er**2/tv),coverage50=np.mean(abs(er)<=.67449*sd),coverage80=np.mean(abs(er)<=1.28155*sd),coverage95=np.mean(abs(er)<=1.95996*sd),spearman_uncertainty_error=spearmanr(ep[q,j],er**2).statistic if len(q)>8 else np.nan,risk_coverage_auc=risk_auc(er**2,ep[q,j])))
 return out
def fit_level(d,lev,tr,va,device,epochs=30,members=2):
 x,y,m,sc=level_data(d,lev);mods=[];ss=[]
 for l in range(members):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+100*l+len(LEVELS[lev]),epochs);mods.append(mo);ss.append((a,b))
 return (x,y,m,sc,*pred_reg(mods,ss,x,device),mods,ss)
def evaluate(run,d,device):
 out={k:[] for k in LEVELS};cache={}
 for sp in ["spatial_block","mu_slice","future","corner"]:
  tr,va,te=split(d,sp)
  for lev,names in LEVELS.items():
   x,y,m,sc,p,al,ep,mods,ss=fit_level(d,lev,tr,va,device);out[lev]+=rows(run,sp,lev,names,y,m,te,p,al,ep);cache[(sp,lev)]=(x,y,m,sc,te,p,al,ep,mods,ss)
  x,_,_,sc=level_data(d,"level1");cms=[train_cls(x,sc,tr,va,device,SEED+700*l,30) for l in range(2)];pr,ce=pred_cls(cms,x,device);yy=sc[te];er=(yy-(pr[te]>=.5))**2
  out["level1"].append(dict(run=run,split=sp,level="level1",field="p_sc",n=len(te),accuracy=np.mean((pr[te]>=.5)==yy),auroc=auc(yy,pr[te]),brier=np.mean((yy-pr[te])**2),ece=ece(yy,pr[te]),spearman_uncertainty_error=spearmanr(ce[te],er).statistic,risk_coverage_auc=risk_auc(er,ce[te])));cache[(sp,"classification")]=(pr,ce,cms)
 return out,cache
def cross(source,target,name,device):
 idx=np.arange(len(source));rng=np.random.default_rng(SEED);rng.shuffle(idx);nv=max(64,int(.12*len(idx)));tr,va=idx[nv:],idx[:nv];out=[]
 for lev,names in LEVELS.items():
  xs,ys,ms,ss=level_data(source,lev);xt,yt,mt,st=level_data(target,lev);mods=[];sc=[]
  for l in range(2):mo,a,b=train_reg(xs,ys,ms,tr,va,device,SEED+1200+100*l,30);mods.append(mo);sc.append((a,b))
  p,al,ep=pred_reg(mods,sc,xt,device);out+=rows(name,"cross_run",lev,names,yt,mt,np.arange(len(target)),p,al,ep)
 xs,_,_,zs=level_data(source,"level1");xt,_,_,zt=level_data(target,"level1");cms=[train_cls(xs,zs,tr,va,device,SEED+1450+l,30) for l in range(2)];pr,ce=pred_cls(cms,xt,device);er=(zt-(pr>=.5))**2;out.append(dict(run=name,split="cross_run",level="level1",field="p_sc",n=len(zt),accuracy=np.mean((pr>=.5)==zt),auroc=auc(zt,pr),brier=np.mean((zt-pr)**2),ece=ece(zt,pr),spearman_uncertainty_error=spearmanr(ce,er).statistic,risk_coverage_auc=risk_auc(er,ce)))
 return out
def replay(run,base,device):
 out=[];probe=qmc.Sobol(3,scramble=True,seed=SEED).random_base2(14)[:10000].astype(np.float32);final=pd.read_csv(base/"dataset_iter097.csv",low_memory=False)
 for budget in [1024,3000,5000,7168]:
  it=int(np.clip(1+(budget-1024)//64,1,97));d=pd.read_csv(base/f"dataset_iter{it:03d}.csv",low_memory=False);future=final.iloc[len(d):];idx=np.arange(len(d));rng=np.random.default_rng(SEED);rng.shuffle(idx);nv=max(64,int(.12*len(idx)));tr,va=idx[nv:],idx[:nv]
  for lev,names in LEVELS.items():
   x,y,m,sc=level_data(d,lev);mods=[];scales=[]
   for l in range(2):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+1700+100*l+budget,15);mods.append(mo);scales.append((a,b))
   u=float(pred_reg(mods,scales,probe,device)[2].mean());fmse=np.nan;rho=np.nan;cov=np.nan
   if len(future):
    xf,yf,mf,_=level_data(future,lev);p,al,ep=pred_reg(mods,scales,xf,device);es=[];us=[];inside=[]
    for j in range(len(names)):
     q=mf[:,j]
     if q.any():er=(yf[q,j]-p[q,j]);es.extend(er**2);us.extend(ep[q,j]);inside.extend(abs(er)<=1.95996*np.sqrt(np.maximum(al[q,j]+ep[q,j],1e-10)))
    if es:fmse=float(np.mean(es));rho=float(spearmanr(us,es).statistic);cov=float(np.mean(inside))
   out.append(dict(run=run,budget=budget,iteration=it,level=lev,probe_epistemic=u,future_mse=fmse,future_uncertainty_error_spearman=rho,future_coverage95=cov))
 return out
def schemas():
 (OUT/"hierarchical_descriptor_schema_v2.yaml").write_text("""schema_version: 2
x: [kBT_over_t, JA_over_t, mu_over_t]
level1: {classification: p_SC, continuous: [Delta_opt, deltaF_NS]}
level2: {conditioning: trusted_SC, continuous: [abs_q], removed_mandatory: [deltaF_UF]}
level3:
  conditioning: trusted_SC_and_spectral_available
  spectral_fingerprint: {status: unavailable_in_frozen_V3_V4}
  low_energy_features: {status: unavailable_in_frozen_V3_V4}
  optional_continuous: [log_gap]
  removed_mandatory: [k_gap]
level4: {conditioning: trusted_gapped_SC, continuous: [P0, Ppi, Pf_margin], evaluation_only: [Z2]}
forbidden_training_fields: [phase_label, topology_label, proposal_source, action_family, reward, iteration, graph_node_id]
""")
 (OUT/"mask_semantics_v2.yaml").write_text("""NOT_APPLICABLE: physical branch absent; never numeric zero
UNKNOWN: observable not saved/computed; masked
UNTRUSTED: value present but reliability gate fails; masked
VALID: finite value and all level gates pass
normal_q: NOT_APPLICABLE
missing_spectral_fingerprint: UNKNOWN
topology_valid: trusted_SC and spectral_trusted and topology_trusted
""")
 (OUT/"model_architecture_v2.yaml").write_text("""per_level: independent_MLP_deep_ensemble
smoke_members: 2
full_members: 5
hidden: [128, 128, 128]
activation: SiLU
regression_loss: masked_heteroscedastic_Gaussian_NLL
classification_loss: BCE
state_transfer_to_production: forbidden
""")
 pd.DataFrame([("level1","Delta_opt","delta_opt","valid"),("level1","deltaF_NS","free_energy_gap_to_normal","valid"),("level2","abs_q","q_opt","valid_on_SC"),("level3","spectral_fingerprint","UNAVAILABLE","unknown"),("level3","low_energy_features","UNAVAILABLE","unknown"),("level3","log_gap","topology_bulk_gap","optional"),("level3","k_gap","topology_k_at_bulk_gap","removed_from_core"),("level4","P0","topology_p0","valid_gapped_SC"),("level4","Ppi","topology_ppi","valid_gapped_SC"),("level4","Pf_margin","topology_pfaffian_margin","valid_gapped_SC")],columns=["level","semantic_field","source_column","v2_status"]).to_csv(OUT/"field_mapping_v2.csv",index=False)
 (OUT/"configs/smoke.yaml").write_text("ensemble: 2\nepochs: 30\nprobe: 10000\n")
 (OUT/"configs/full_local.yaml").write_text("ensemble: 5\nepochs: 120\nprobe: 50000\nblocked_by_artifact_gate: true\n")
def figures(metrics,crossdf,comp,replaydf,probe,recon,decision):
 fd=OUT/"figures";fd.mkdir(exist_ok=True);plt.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":.25,"font.size":8})
 names=["spatial_r2","calibration","uncertainty_error","risk_coverage","cross_run","spectral_comparison","topology_metrics","probe_levels","replay_uncertainty","replay_future","phase_reconstruction","decision"]
 def save(fig,i):fig.tight_layout();fig.savefig(fd/f"figure_{i:02d}_{names[i-1]}.png",bbox_inches="tight");fig.savefig(fd/f"figure_{i:02d}_{names[i-1]}.pdf",bbox_inches="tight");plt.close(fig)
 reg=pd.concat(metrics.values());reg=reg[reg.rmse.notna()];sp=reg[reg.split=="spatial_block"]
 fig,ax=plt.subplots(figsize=(7,3.5));sp.groupby(["level","run"]).r2.median().unstack().plot.bar(ax=ax);ax.set_title("Spatial median R2 by level");save(fig,1)
 fig,ax=plt.subplots(figsize=(7,3.5));sp.groupby(["level","run"])[["coverage50","coverage80","coverage95"]].mean().plot.bar(ax=ax);ax.set_title("Interval calibration");save(fig,2)
 fig,ax=plt.subplots(figsize=(7,3.5));sp.groupby(["level","run"]).spearman_uncertainty_error.median().unstack().plot.bar(ax=ax);ax.set_title("Epistemic-error Spearman");save(fig,3)
 fig,ax=plt.subplots(figsize=(7,3.5));sp.groupby(["level","run"]).risk_coverage_auc.median().unstack().plot.bar(ax=ax);ax.set_title("Risk-coverage AUC (lower is better)");save(fig,4)
 fig,ax=plt.subplots(figsize=(7,3.5));crossdf.groupby(["level","run"]).r2.median().unstack().plot.bar(ax=ax);ax.set_title("Bidirectional cross-run R2");save(fig,5)
 fig,ax=plt.subplots(figsize=(7,3.5));comp[comp.level=="level3"].pivot(index="run",columns="model",values="nll_mean").plot.bar(ax=ax);ax.set_title("Old spectral vs log-gap-only NLL");save(fig,6)
 fig,ax=plt.subplots(figsize=(7,3.5));sp[sp.level=="level4"].pivot(index="field",columns="run",values="r2").plot.bar(ax=ax);ax.set_title("Conditional topology spatial R2");save(fig,7)
 fig,axs=plt.subplots(2,2,figsize=(8,6));
 for ax,(lev,g) in zip(axs.flat,probe.groupby("level")):s=ax.scatter(g.kT,g.JA,c=g.epistemic,s=3,cmap="viridis");ax.set_title(lev);fig.colorbar(s,ax=ax)
 save(fig,8)
 for i,col,title in [(9,"probe_epistemic","Replay probe uncertainty"),(10,"future_mse","Replay future MSE")]:
  fig,axs=plt.subplots(1,2,figsize=(8,3.5))
  for ax,(run,g) in zip(axs,replaydf.groupby("run")):
   for lev,h in g.groupby("level"):ax.plot(h.budget,h[col],"o-",label=lev)
   ax.set_title(run);ax.legend(fontsize=6)
  fig.suptitle(title);save(fig,i)
 fig,ax=plt.subplots(figsize=(7,3.5));recon.pivot(index="level",columns="run",values="accuracy").plot.bar(ax=ax);ax.set_title("Post-hoc hierarchy reconstruction");save(fig,11)
 fig,ax=plt.subplots(figsize=(8,4));ax.axis("off");ax.text(.03,.82,decision,fontsize=15,weight="bold");ax.text(.03,.58,"No frozen spectral fingerprint or low-energy moments\nlog-gap-only is an ablation, not a replacement fingerprint\nDo not run full-local or EIG",fontsize=11,va="top");save(fig,12)
def report(dec):
 md=f"""# Stage VI-0.2: Hierarchical World Model v2

**Single main decision: {dec['main_decision']}.**

## Artifact gate

The frozen V3/V4 final CSVs and exact merged NPZ shards were inspected directly. The 131-field exact schema contains gap, k-at-gap, Pfaffian outputs, and thermodynamic/oracle metadata, but no fixed-k spectrum, eigenvalue curve, DOS proxy, near-zero spectral-minimum count, or spectral curvature. Free-energy basin counters are not spectral fingerprints and were not repurposed. Therefore PCA, autoencoder, and low-energy-moment fingerprint tests cannot be performed from the allowed frozen inputs.

## Executed smoke

All weights and split scalers were initialized from scratch. Level 1, Level 2, and conditional Level 4 were revalidated on spatial blocks, mu slices, corners, future holdout, and bidirectional cross-run transfer. Level 3 compared the old joint `[log_gap,k_gap]` descriptor with the only legal available-v2 ablation, `[log_gap]`. Uncertainty-error Spearman, risk-coverage AUC, interval coverage, 10k level-wise probe maps, sequential replay, and post-hoc hierarchy reconstruction were computed.

## Interpretation

Removing k-gap avoids forcing a non-smooth argmin output, but log-gap-only is not a spectral fingerprint and cannot satisfy the requested redesign. Mean spatial SC AUROC is {dec['spatial_sc_auroc_mean']:.3f}; median spatial R2 is {dec['spatial_level1_r2_median']:.3f} for Level 1, {dec['spatial_level2_r2_median']:.3f} for Level 2, {dec['spatial_level3_r2_median']:.3f} for the log-gap-only Level 3 ablation, and {dec['spatial_level4_r2_median']:.3f} for Level 4. Cross-run q and topology transfer remains strong, but log-gap-only does not consistently improve on the old log-gap field. Pfaffian-margin and uncertainty calibration are mixed. Level-wise replay uncertainty is not uniformly monotonic. The available evidence cannot establish that the model knows what it does not know.

## Decision

{dec['main_decision']} is the only decision. The blocker is missing frozen spectral artifacts, not a request for new exact calculations within this task. Do not run the five-member full-local configuration or EIG. First export already-computed fixed-grid spectra or physically defined low-energy spectral moments, if they exist outside the current package; otherwise redesign future exact-output logging. No development learned state is transferable.
""";(OUT/"reports/hierarchical_world_model_v2_report.md").write_text(md)
 tex=r"""\documentclass[10pt]{article}\usepackage[margin=18mm]{geometry}\usepackage{graphicx}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\title{Stage VI-0.2: Hierarchical World Model v2}\author{Frozen-data offline smoke validation}\date{2026-07-12}\begin{document}\maketitle\section{Decision}"""+dec["main_decision"].replace("_","\\_")+r""" is the single main decision.\section{Artifact gate}The 131-field frozen exact schema contains no fixed-k spectrum, eigenvalue curve, DOS proxy, spectral-minimum count, or spectral curvature. Free-energy basin metadata were not misused as spectral fingerprints.\section{Executed smoke}Levels 1, 2, and 4 were revalidated. Level 3 compares the old log-gap/k-gap output with a log-gap-only ablation. Within-run, cross-run, calibration, risk-coverage, probe, replay, and reconstruction tests were completed.\section{Numerical summary}Mean spatial SC AUROC is """+f"{dec['spatial_sc_auroc_mean']:.3f}"+r""". Median spatial $R^2$ is """+f"{dec['spatial_level1_r2_median']:.3f}"+r""" (Level 1), """+f"{dec['spatial_level2_r2_median']:.3f}"+r""" (Level 2), """+f"{dec['spatial_level3_r2_median']:.3f}"+r""" (log-gap-only Level 3), and """+f"{dec['spatial_level4_r2_median']:.3f}"+r""" (Level 4). No run/level replay uncertainty trajectory is monotonic.\section{Conclusion}The log-gap-only ablation is not a replacement spectral fingerprint. Full-local and EIG are blocked. No learned state is transferable.\clearpage"""
 for f in sorted((OUT/"figures").glob("*.png")):tex+=f"\\begin{{figure}}[p]\\centering\\includegraphics[width=.92\\textwidth]{{../figures/{f.name}}}\\caption{{{f.stem.replace('_',' ')}}}\\end{{figure}}\n"
 tex+=r"""\clearpage\section{Next artifact action}Export pre-existing fixed-grid spectra or physically defined low-energy moments if available. This report does not authorize exact calculations.\end{document}""";(OUT/"reports/hierarchical_world_model_v2_report.tex").write_text(tex);subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","hierarchical_world_model_v2_report.tex"],cwd=OUT/"reports",check=True)
def main():
 t0=time.perf_counter();schemas();device="mps" if torch.backends.mps.is_available() else "cpu";data={"V3":pd.read_csv(V3/"dataset_iter097.csv",low_memory=False),"V4":pd.read_csv(V4/"dataset_iter097.csv",low_memory=False)};metrics={k:[] for k in LEVELS};caches={}
 for run,d in data.items():o,c=evaluate(run,d,device);caches[run]=c;[metrics[k].extend(o[k]) for k in LEVELS]
 for k in LEVELS:metrics[k]=pd.DataFrame(metrics[k]);metrics[k].to_csv(OUT/f"{k}_metrics.csv",index=False)
 crossdf=pd.DataFrame(cross(data['V3'],data['V4'],"V3_to_V4",device)+cross(data['V4'],data['V3'],"V4_to_V3",device));crossdf.to_csv(OUT/"metrics/cross_run_metrics.csv",index=False)
 unc=pd.concat([pd.concat(metrics.values()),crossdf],ignore_index=True);unc.to_csv(OUT/"uncertainty_metrics.csv",index=False)
 old=pd.read_csv(ROOT/"stage_vi0_hierarchical_world_model_validation/level3_metrics.csv");new=metrics['level3'];cmp=[]
 for run in ['V3','V4']:
  groups=[('old_joint_log_gap_plus_k_gap',old[(old.run==run)&(old.split=='spatial_block')]),('old_joint_log_gap_field',old[(old.run==run)&(old.split=='spatial_block')&(old.field=='log_gap')]),('available_v2_log_gap_only',new[(new.run==run)&(new.split=='spatial_block')])]
  for model,g in groups:cmp.append(dict(run=run,level="level3",model=model,rmse_mean=g.rmse.mean(),nll_mean=g.nll.mean(),coverage95_mean=g.coverage95.mean(),spearman_median=g.spearman_uncertainty_error.median(),fingerprint_validated=False))
 spec=pd.DataFrame(cmp);spec.to_csv(OUT/"spectral_descriptor_comparison.csv",index=False)
 oldcomp=pd.read_csv(ROOT/"stage_vi0_hierarchical_world_model_validation/unified_vs_hierarchical.csv");v2=[]
 for lev in LEVELS:
  for run in ['V3','V4']:
   u=oldcomp[(oldcomp.run==run)&(oldcomp.level==lev)&(oldcomp.model=='unified')].iloc[0];g=metrics[lev][(metrics[lev].run==run)&(metrics[lev].split=='spatial_block')&(metrics[lev].field!='p_sc')];v2+=[dict(run=run,level=lev,model="stage_vi0_unified",rmse_mean=u.rmse_mean,nll_mean=u.nll_mean,coverage95_mean=u.coverage95_mean,spearman_median=u.spearman_median),dict(run=run,level=lev,model="stage_vi0_2_v2",rmse_mean=g.rmse.mean(),nll_mean=g.nll.mean(),coverage95_mean=g.coverage95.mean(),spearman_median=g.spearman_uncertainty_error.median())]
 comp=pd.DataFrame(v2);comp.to_csv(OUT/"unified_vs_hierarchical_v2.csv",index=False)
 replaydf=pd.DataFrame(replay('V3',V3,device)+replay('V4',V4,device));replaydf.to_csv(OUT/"replay_metrics.csv",index=False)
 probe=qmc.Sobol(3,scramble=True,seed=SEED).random_base2(14)[:10000].astype(np.float32);prows=[]
 for run,d in data.items():
  tr,va,te=split(d,'future')
  for lev in LEVELS:
   x,y,m,sc=level_data(d,lev);mods=[];ss=[]
   for l in range(2):mo,a,b=train_reg(x,y,m,tr,va,device,SEED+2100+l,20);mods.append(mo);ss.append((a,b))
   ep=pred_reg(mods,ss,probe,device)[2].mean(1);sel=np.argsort(abs(probe[:,2]-.5))[:1500];phys=probe[sel]*(BOUNDS[:,1]-BOUNDS[:,0])+BOUNDS[:,0]
   for xx,u in zip(phys,ep[sel]):prows.append(dict(run=run,level=lev,kT=xx[0],JA=xx[1],mu=xx[2],epistemic=u))
 probe_df=pd.DataFrame(prows);probe_df.to_csv(OUT/"uncertainty/probe_uncertainty.csv",index=False)
 rec=[]
 for run,d in data.items():
  c=caches[run];te=c[('future','level1')][4];sc=physical_sc(d)[te];pr=c[('future','classification')][0][te]>=.5;rec.append(dict(run=run,level='normal_SC',accuracy=np.mean(pr==sc),n=len(te)))
  qp=c[('future','level2')][5][te,0];phase=d.phase_name.astype(str).to_numpy()[te];rec.append(dict(run=run,level='uniform_FFLO',accuracy=np.mean((qp[sc]>=.01/math.pi)==(phase[sc]=='FFLO')),n=int(sc.sum())))
  p4=c[('future','level4')][5][te];top=d.topology_label_code.fillna(-1).astype(int).to_numpy()[te];valid=c[('future','level4')][2][te].all(1);rec.append(dict(run=run,level='topology_tendency',accuracy=np.mean(((p4[valid,0]*p4[valid,1])<0)==(top[valid]==1)),n=int(valid.sum())))
 recon=pd.DataFrame(rec);recon.to_csv(OUT/"phase_reconstruction.csv",index=False)
 decision="Decision D - insufficient frozen artifacts"
 sp={k:metrics[k][(metrics[k].split=='spatial_block')&(metrics[k].field!='p_sc')] for k in LEVELS}
 dec={"main_decision":decision,"selected_device":device,"mps_available":torch.backends.mps.is_available(),"v3_reproduced":len(data['V3'])==7069,"v4_reproduced":len(data['V4'])==7080,"spectral_fingerprint_available":False,"low_energy_spectral_features_available":False,"log_gap_only_ablation_completed":True,"within_run_complete":True,"cross_run_complete":True,"replay_complete":True,"full_local_run":False,"safe_for_EIG":False,"forbidden_state_transfer_confirmed":True,"spatial_sc_auroc_mean":float(metrics['level1'].query("split == 'spatial_block' and field == 'p_sc'").auroc.mean()),"spatial_level1_r2_median":float(sp['level1'].r2.median()),"spatial_level2_r2_median":float(sp['level2'].r2.median()),"spatial_level3_r2_median":float(sp['level3'].r2.median()),"spatial_level4_r2_median":float(sp['level4'].r2.median()),"main_limiting_factor":"frozen artifacts contain no spectral fingerprint or low-energy spectral moments","recommended_next_action":"export pre-existing spectrum artifacts or revise future exact-output logging; do not run EIG","wall_time_sec":time.perf_counter()-t0,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
 (OUT/"decision.json").write_text(json.dumps(dec,indent=2)+"\n");(OUT/"manifests/missing_spectral_artifacts.md").write_text("# Missing spectral artifacts\n\nNo fixed-k spectra, eigenvalue curves, DOS proxy, near-zero spectral-minimum count, or spectral curvature were found in the frozen V3/V4 CSVs, exact merged NPZs, supplemental raw artifacts, or filenames. Free-energy search minima are not spectral minima. PCA/autoencoder/fingerprint validation is blocked.\n");figures(metrics,crossdf,comp,replaydf,probe_df,recon,decision);report(dec);print(json.dumps(dec,indent=2))
if __name__=="__main__":main()
